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
from kauldron.train import config_lib
from kauldron.train import train_step
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member

# pylint: disable=logging-fstring-interpolation

# XManager API do not have API for jobs within a work-unit to communicate,
# so use files for communication.
TRAIN_COMPLETE_FILENAME = 'train_complete.txt'


def continuous_eval(
    trainer: config_lib.Trainer,
    eval_names: list[str],
    final_eval_names: list[str],
) -> dict[str, train_step.Auxiliaries]:
  """Continuous evaluation.

  Trigger an evaluation everytime a new checkpoint is detected.

  Args:
    trainer: Trainer to evaluate
    eval_names: Eval names to run.
    final_eval_names: Eval names to run after the training is complete.

  Returns:
    Auxiliaries: Dict eval name -> last auxiliaries

  Raises:
    Exception: Re-raises any exception thrown by underlying evaluators.
  """
  # Validation
  for eval_name in eval_names:
    if eval_name not in trainer.evals:
      raise ValueError(f'Invalid eval name. Available: {list(trainer.evals)}')

  logging.info('Initialize the state...')
  ckpt = trainer.checkpointer
  # Skip transforms as checkpoint is restored anyway afterward. We need to
  # be careful that step 0 is indeed computed from the checkpoint.
  state = trainer.init_state(skip_transforms=True)
  aux = {eval_name: train_step.Auxiliaries() for eval_name in eval_names}

  # If preempted, the last checkpoint might be re-computed. There could be
  # some race condition where the metrics are written twice for one step, but
  # likely not an issue in practice.

  # Rather than failing if a single eval fails, we keep track of all failures
  # and raise them at the end.
  tracker = _ExceptionTracker(eval_names=list(eval_names))  # `list` as mutated

  logging.info('Start evaluating...')
  final_step = 0
  for step in ckpt.iter_new_checkpoints(
      min_interval_secs=10,
      timeout=10,
      # Check train is complete
      timeout_fn=lambda: epath.Path(trainer.workdir)
      .joinpath(TRAIN_COMPLETE_FILENAME)
      .exists(),
  ):
    logging.info(f'Processing checkpoint for step {step}...')

    # Refresh the checkpoint manager cache used by `.all_steps()`
    # (b/315316885#6)
    ckpt.reload()

    state = ckpt.restore(state, step=step)
    assert int(state.step) == step

    # Processing checkpoint
    aux = dict()
    for name in eval_names:
      with tracker.catch_exception(name=name, step=step):
        aux[name] = trainer.evals[name].evaluate(state=state, step=step)

    final_step = step

  logging.info('Running final evals...')
  if final_eval_names:
    aux = dict()
  for name in final_eval_names:
    with tracker.catch_exception(name=name, step=final_step):
      aux[name] = trainer.evals[name].evaluate(state=state, step=final_step)

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
