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

from absl import logging
from etils import exm
from kauldron.train import config_lib
from kauldron.train import train_step

# pylint: disable=logging-fstring-interpolation

# XManager API do not have API for jobs within a work-unit to communicate,
# so use work-units tags.
TRAIN_COMPLETE_TAG = 'train_complete'


def continuous_eval(
    trainer: config_lib.Trainer,
    eval_names: list[str],
) -> dict[str, train_step.Auxiliaries]:
  """Continuous evaluation.

  Trigger an evaluation everytime a new checkpoint is detected.

  Args:
    trainer: Trainer to evaluate
    eval_names: Eval names to run.

  Returns:
    Auxiliaries: Dict eval name -> last auxiliaries
  """
  # Validation
  for eval_name in eval_names:
    if eval_name not in trainer.evals:
      raise ValueError(f'Invalid eval name. Available: {list(trainer.evals)}')

  logging.info('Initialize the state...')
  ckpt = trainer.checkpointer
  state = trainer.init_state()
  aux = {eval_name: train_step.Auxiliaries() for eval_name in eval_names}

  # TODO(epot): Checkpoint should save the state ? Otherwise, the last
  # checkpoint might be re-computed if prehempted ?

  logging.info('Start evaluating...')
  for step in ckpt.iter_new_checkpoints(
      min_interval_secs=10,
      timeout=10,
      timeout_fn=_is_train_complete,
  ):
    logging.info(f'Processing checkpoint for step {step}...')

    # Due to b/315316885, we need to explicitly pass the `step` to restore,
    # as `latest_step` is cached.
    ckpt.refresh_cache()

    state = ckpt.restore(state, step=step, noop_if_missing=True)
    assert int(state.step) == step

    # Processing checkpoint
    aux = {
        eval_name: trainer.evals[eval_name].evaluate(state, step=step)
        for eval_name in eval_names
    }

  # Return the last aux
  return aux


def _is_train_complete() -> bool:
  """Detect if `train` XManager job is complete."""
  wu = exm.current_work_unit()
  wu.refresh()  # Refresh the tags
  return TRAIN_COMPLETE_TAG in wu.tags
