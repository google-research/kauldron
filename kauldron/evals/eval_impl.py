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

from absl import logging
from etils import epath
from kauldron.evals import evaluators as evaluators_lib
from kauldron.evals import run_strategies
from kauldron.train import train_step
from kauldron.train import trainer_lib
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

# pylint: disable=logging-fstring-interpolation

# XManager API do not have API for jobs within a work-unit to communicate,
# so use files for communication.
TRAIN_COMPLETE_FILENAME = 'train_complete.txt'
EVAL_COMPLETE_FILENAME = 'eval_{}_complete.txt'


def continuous_eval(
    trainer: trainer_lib.Trainer,
    eval_names: list[str],
) -> dict[str, train_step.Auxiliaries]:
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
  ckpt = trainer.checkpointer
  # Skip transforms as checkpoint is restored anyway afterward. We need to
  # be careful that step 0 is indeed computed from the checkpoint.
  # In eval-only mode, the model weights are restored from the init_transforms
  # and not the checkpoint, so we cannot skip it.
  state = trainer.init_state(skip_transforms=not trainer.setup.eval_only)
  aux = {eval_name: train_step.Auxiliaries() for eval_name in eval_names}

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
  # the step from the `init_transforms`.
  final_step = int(state.step)
  for step in ckpt.iter_new_checkpoints(
      min_interval_secs=10,
      timeout=10,
      timeout_fn=lambda: (
          # Skip the `iter_new_checkpoints` for eval-only jobs.
          trainer.setup.eval_only
          # Exit when train job has completed
          or epath.Path(trainer.workdir)
          .joinpath(TRAIN_COMPLETE_FILENAME)
          .exists()
      ),
  ):
    logging.info(f'Processing checkpoint for step {step}...')

    state = ckpt.restore(state, step=step)
    assert int(state.step) == step

    # Processing checkpoint
    aux = dict()
    for ev in every_checkpoint_evals:
      with tracker.catch_exception(name=ev.name, step=step):
        aux[ev.name] = ev.evaluate(state=state, step=step)

    final_step = step

  # All every_checkpoint_evals have been processed. Marks those as complete.
  if trainer.workdir.exists():  # `TrainEvaluator` do not have a workdir
    for ev in every_checkpoint_evals:
      epath.Path(trainer.workdir).joinpath(
          EVAL_COMPLETE_FILENAME.format(ev.name)
      ).touch()

  logging.info('Running final evals...')
  for ev in last_checkpoint_evals:
    with tracker.catch_exception(name=ev.name, step=final_step):
      aux[ev.name] = ev.evaluate(state=state, step=final_step)

  # All last_checkpoint_evals have been processed. Marks those as complete.
  if trainer.workdir.exists():  # `TrainEvaluator` do not have a workdir
    for ev in last_checkpoint_evals:
      epath.Path(trainer.workdir).joinpath(
          EVAL_COMPLETE_FILENAME.format(ev.name)
      ).touch()

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
