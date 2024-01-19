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

import time

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
  state = trainer.init_state()
  aux = {eval_name: train_step.Auxiliaries() for eval_name in eval_names}

  logging.info('Start evaluating...')
  last_step = -1
  while True:
    # First check if training is complete to avoid a race condition (training
    # finishing before the last checkpoint is processed)
    train_complete = _is_train_complete()

    trainer.checkpointer.refresh_cache()
    # Optimization: Could detect the last step without restoring the checkpoint
    state = trainer.checkpointer.restore(state, noop_if_missing=True)

    # Detect whether the checkpoint should be processed or not
    restored_step = int(state.step)
    if restored_step == last_step:
      if train_complete:  # Already processed the last step
        logging.info(
            f'All checkpoints processed (last: {last_step}). Exiting...'
        )
        break
      else:
        logging.info(f'Waiting for a new checkpoint (last: {last_step})...')
        time.sleep(10)
        continue
    logging.info(f'Processing checkpoint for step {restored_step}...')

    # Processing checkpoint
    aux = {
        eval_name: trainer.evals[eval_name].evaluate(state, step=restored_step)
        for eval_name in eval_names
    }
    last_step = restored_step

  # Return the last aux
  return aux


def _is_train_complete() -> bool:
  """Detect if `train` XManager job is complete."""
  wu = exm.current_work_unit()
  wu.refresh()  # Refresh the tags
  return TRAIN_COMPLETE_TAG in wu.tags
