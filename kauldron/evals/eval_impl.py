# Copyright 2024 The kauldron Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`Trainer.eval` implementation."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import dataclasses
import hashlib

from absl import logging
from etils import epath
from kauldron.checkpoints import checkpointer
from kauldron.checkpoints import partial_loader
from kauldron.evals import evaluators as evaluators_lib
from kauldron.evals import run_strategies
from kauldron.train import auxiliaries
from kauldron.train import train_step
from kauldron.train import trainer_lib
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member
import orbax.checkpoint as ocp

# pylint: disable=logging-fstring-interpolation

# XManager API do not have API for jobs within a work-unit to communicate,
# so use files for communication.
TRAIN_COMPLETE_FILENAME = 'train_complete.txt'


def continuous_eval(
    trainer: trainer_lib.Trainer,
    eval_names: list[str],
) -> dict[str, auxiliaries.AuxiliariesState]:
  """Continuous evaluation.

  Trigger an evaluation everytime a new checkpoint is detected.

  Args:
    trainer: Trainer to evaluate
    eval_names: Eval names to run.

  Returns:
    Auxiliaries: Dict eval name -> last auxiliaries

  Raises:
    Exception: Re-raises any exception thrown by underlying evaluators.
  """
  trainer.setup.run(trainer)

  # Validation
  for eval_name in eval_names:
    if eval_name not in trainer.evals:
      raise ValueError(f'Invalid eval name. Available: {list(trainer.evals)}')

  logging.info('Initialize the state...')
  # Skip transforms as checkpoint is restored anyway afterward. We need to
  # be careful that step 0 is indeed computed from the checkpoint.
  # In eval-only mode, the model weights are restored from the init_transform
  # and not the checkpoint, so we cannot skip it.
  state = trainer.init_state(
      skip_transforms=not trainer.setup.eval_only,
      skip_optimizer=all(
          trainer.evals[name].discard_opt for name in eval_names
      ),
  )
  aux = {eval_name: auxiliaries.AuxiliariesState() for eval_name in eval_names}

  # If preempted, the last checkpoint might be re-computed. There could be
  # some race condition where the metrics are written twice for one step, but
  # likely not an issue in practice.

  # Rather than failing if a single eval fails, we keep track of all failures
  # and raise them at the end.
  tracker = _ExceptionTracker(eval_names=list(eval_names))  # `list` as mutated

  # Split evaluators
  every_checkpoint_evals: list[evaluators_lib.EvaluatorBase] = []
  last_checkpoint_evals: list[evaluators_lib.EvaluatorBase] = []
  for name in eval_names:
    ev = trainer.evals[name]
    if isinstance(ev.run, run_strategies.StandaloneLastCheckpoint):
      last_checkpoint_evals.append(ev)
    elif isinstance(ev.run, run_strategies.StandaloneEveryCheckpoint):
      every_checkpoint_evals.append(ev)
    else:
      raise ValueError(
          f'Remote eval ({name!r}) should be standalone. Got run={ev.run}'
      )

  logging.info('Start evaluating...')
  # Initialize the final step from the state for eval-only jobs which restore
  # the step from the `init_transform`.
  final_step = int(state.step)
  for state in _preemptable_iter_new_checkpoints(
      trainer=trainer,
      eval_names=eval_names,
      state=state,
  ):
    step = int(state.step)
    logging.info(f'Processing checkpoint for step {step}...')
    # Processing checkpoint
    aux = dict()
    for ev in every_checkpoint_evals:
      with tracker.catch_exception(name=ev.name, step=step):
        aux[ev.name] = ev.evaluate(state=state, step=step)
    final_step = step

  logging.info('Running final evals...')
  for ev in last_checkpoint_evals:
    with tracker.catch_exception(name=ev.name, step=final_step):
      aux[ev.name] = ev.evaluate(state=state, step=final_step)

  tracker.maybe_reraise()

  # Return the last aux
  return aux


@dataclasses.dataclass
class _ExceptionTracker:
  """Context manager to track exceptions."""

  eval_names: list[str]
  exceptions: list[Exception] = dataclasses.field(default_factory=list)

  @contextlib.contextmanager
  def catch_exception(
      self,
      *,
      name: str,
      step: int,
  ) -> Iterator[None]:
    """Context manager which record exceptions."""
    try:
      yield
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.exceptions.append(e)
      logging.exception('Failed to evaluate %s at step %s', name, step)
      exc_name = type(e).__name__
      status.xp.add_tags(f'🚨 Eval {name}: {exc_name} 🚨')
      self.eval_names.remove(name)
    if not self.eval_names:
      # All evaluator have failed, re-raise the exception
      raise ExceptionGroup('All evaluators have failed', self.exceptions)

  def maybe_reraise(self) -> None:
    if self.exceptions:
      raise ExceptionGroup('One or more evaluators failed', self.exceptions)


def _preemptable_iter_new_checkpoints(
    *,
    trainer: trainer_lib.Trainer,
    eval_names: list[str],
    state: train_step.TrainState,
) -> Iterator[train_step.TrainState]:
  """Yields the new checkpoints."""
  # Skip the `iter_new_checkpoints` for eval-only jobs.
  if 'save_tmp_ckpt' in trainer.aux:
    save_tmp_ckpt = trainer.aux['save_tmp_ckpt']
  else:
    save_tmp_ckpt = True

  if trainer.setup.eval_only:
    return

  trainer_ckpt = trainer.checkpointer
  assert isinstance(trainer_ckpt, checkpointer.Checkpointer)
  with _get_eval_ckpt(trainer_ckpt, eval_names) as eval_ckpt:
    # If the eval checkpoint exists, there is an ongoing eval that was preempted
    # and we should resume the onging eval.
    # After the eval is done, we check for new checkpoints and write a new
    # checkpoint, which then will be picked up if the job gets preempted again.
    if eval_ckpt.latest_step is not None:
      logging.info('Resume evaluation...')
      # Restore the state from the last eval checkpoint
      state = eval_ckpt.restore(state)
      step = int(state.step)
      yield state
      # state might have been donated, we should not access it after this point.
      # Eval is done, remove the duplicated checkpoint
      eval_ckpt.delete(step)
    for step in trainer_ckpt.iter_new_checkpoints(
        min_interval_secs=10,
        timeout=10,
        timeout_fn=lambda: (
            # Exit when train job has completed
            epath.Path(trainer.workdir)
            .joinpath(TRAIN_COMPLETE_FILENAME)
            .exists()
        ),
    ):
      state = _restore_checkpoint(
          trainer_ckpt=trainer_ckpt,
          state=state,
          step=step,
      )
      assert int(state.step) == step
      # Temporarily copy the state to the eval checkpoint, to ensure that
      # it won't be deleted by the train job until the current eval is done.
      if save_tmp_ckpt:
        eval_ckpt.save(state, step=step)
      yield state
      # state might have been donated, we should not access it after this point.
      # Eval is done, remove the duplicated checkpoint
      if save_tmp_ckpt:
        eval_ckpt.delete(step)


def _restore_checkpoint(
    *,
    trainer_ckpt: checkpointer.Checkpointer,
    state: train_step.TrainState,
    step: int,
) -> train_step.TrainState:
  """Restores the checkpoint."""
  # TODO(epot): Rather than `PartialKauldronLoader`, should instead
  # have some `trainer_ckpt.restore(state, partial_restore=True)`
  if state.opt_state is None:
    # PartialKauldronLoader restores params and collections
    with partial_loader.PartialKauldronLoader(
        workdir=trainer_ckpt.workdir,
        step=step,
    ) as loader:
      return loader.transform(state)
  return trainer_ckpt.restore(state, step=step)


def _get_eval_ckpt(
    trainer_ckpt: checkpointer.Checkpointer,
    eval_names: list[str],
) -> checkpointer.Checkpointer:
  """Returns the checkpoint to use for the eval."""
  if isinstance(trainer_ckpt, checkpointer.Checkpointer):
    suffix = ':'.join(eval_names)
    # Use hash if suffix is too long.
    if len(suffix) > 100:
      suffix = hashlib.sha256(suffix.encode('utf-8')).hexdigest()[:8]
    workdir = epath.Path(trainer_ckpt.workdir) / f'_eval_{suffix}'
    (workdir / checkpointer.CHECKPOINT_FOLDER_NAME).mkdir(
        parents=True, exist_ok=True
    )
    logging.info('Create checkpointer with: workdir: %s', workdir)
    return checkpointer.Checkpointer(
        workdir=workdir,
        save_interval_steps=1,
        max_to_keep=1,
        create=False,
        # Adding an explicit prefix to avoid collisions with the train-eval job.
        multiprocessing_options=ocp.options.MultiprocessingOptions(
            barrier_sync_key_prefix='eval'
        ),
    )
  else:
    raise ValueError(f'Unsupported checkpointer type: {type(trainer_ckpt)}')
