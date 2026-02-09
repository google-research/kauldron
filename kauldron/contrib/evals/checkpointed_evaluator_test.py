# Copyright 2026 The kauldron Authors.
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

"""Tests for CheckpointedEvaluator and its checkpointer."""

import os
import pathlib

import flax.struct
import jax.numpy as jnp
from kauldron import kd
from kauldron.contrib.evals import checkpointed_evaluator
from examples import mnist_autoencoder


@flax.struct.dataclass
class CountBatches(kd.metrics.Metric):
  count: int

  @classmethod
  def empty(cls) -> 'CountBatches':
    return cls(count=0)

  def update(self, batch) -> 'CountBatches':
    del batch
    return self.replace(count=self.count + 1)

  def merge(self, other: 'CountBatches') -> 'CountBatches':
    return self.replace(count=self.count + other.count)


class StopEvaluator(checkpointed_evaluator.CheckpointedEvaluator):
  """An evaluator that stops with an error at step 3."""

  def step(self, *, step_nr, state, batch):
    if jnp.asarray(step_nr) == 3:
      raise ValueError('Simulated preemption')
    return super().step(step_nr=step_nr, state=state, batch=batch)


def test_checkpointing(tmp_path: pathlib.Path):
  ckpt_dir = tmp_path / 'eval0'
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = os.fspath(tmp_path)
  cfg.eval_ds = cfg.train_ds
  cfg.num_train_steps = 0  # No training

  # 1. Run evaluation with an evaluator that stops at step 3
  with kd.konfig.mock_modules():
    cfg.evals = {
        'stopping_eval': StopEvaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=None,
            metrics={'batch_count': CountBatches(0)},
            checkpointer=kd.checkpoints.Checkpointer(
                workdir=ckpt_dir, save_interval_steps=1, max_to_keep=None
            ),
        )
    }
  cfg1 = kd.konfig.resolve(cfg)
  trainer1 = kd.konfig.resolve(cfg1.trainer)
  state = trainer1.init_state()
  evaluator1 = cfg1.evals['stopping_eval']

  try:
    evaluator1.evaluate(state, 0)
    assert False, 'Should have raised ValueError'
  except ValueError as e:
    assert str(e) == 'Simulated preemption'

  # Check that checkpoint for step 2 exists
  assert (ckpt_dir / '2').exists()
  assert not (ckpt_dir / '3').exists()

  # 2. Run evaluation with a normal CheckpointedEvaluator
  # It should resume from step 2 and finish.
  with kd.konfig.mock_modules():
    cfg.evals = {
        'resuming_eval': checkpointed_evaluator.CheckpointedEvaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=None,
            metrics={'batch_count': CountBatches(0)},
            checkpointer=kd.checkpoints.Checkpointer(
                workdir=ckpt_dir, save_interval_steps=1, max_to_keep=None
            ),
        )
    }
  cfg2 = kd.konfig.resolve(cfg)
  evaluator2 = cfg2.evals['resuming_eval']
  merged_aux = evaluator2.evaluate(state, 0)

  assert merged_aux.metrics['batch_count'].count == 4
  assert (ckpt_dir / '4').exists()


def test_checkpointer_workdir(tmp_path: pathlib.Path):
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = os.fspath(tmp_path)
  cfg.eval_ds = cfg.train_ds
  cfg.num_train_steps = 0  # No training

  # Test with default checkpointer workdir (which defaults to cfg.workdir)
  with kd.konfig.mock_modules():
    cfg.evals = {
        'my_eval': checkpointed_evaluator.CheckpointedEvaluator(
            run=kd.evals.EveryNSteps(1),
            checkpointer=kd.checkpoints.Checkpointer(
                save_interval_steps=1, max_to_keep=None
            ),
        )
    }
  cfg = kd.konfig.resolve(cfg)
  evaluator = cfg.evals['my_eval']

  expected_workdir = tmp_path / 'evals' / 'my_eval'
  assert evaluator.checkpointer.workdir == expected_workdir

  # Test with a custom checkpointer workdir
  custom_ckpt_dir = tmp_path / 'custom_checkpoints'
  with kd.konfig.mock_modules():
    cfg.evals = {
        'my_eval': checkpointed_evaluator.CheckpointedEvaluator(
            run=kd.evals.EveryNSteps(1),
            checkpointer=kd.checkpoints.Checkpointer(
                workdir=custom_ckpt_dir,
                save_interval_steps=1,
                max_to_keep=None,
            ),
        )
    }
  cfg = kd.konfig.resolve(cfg)
  evaluator = cfg.evals['my_eval']
  assert evaluator.checkpointer.workdir == custom_ckpt_dir
