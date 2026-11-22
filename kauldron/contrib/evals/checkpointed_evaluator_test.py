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

import dataclasses
import os
import pathlib
from typing import Any, Optional

import flax.struct
import jax
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


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ConcatAndClearArrayMetric(kd.metrics.Metric):
  """Metric whose State exercises host conversion and clearing array fields to None in finalize()."""

  image: kd.kontext.Key = 'batch.image'

  @flax.struct.dataclass
  class State(kd.metrics.State['ConcatAndClearArrayMetric']):
    raw_batch_mean: Optional[jax.Array] = None
    accumulated_sum: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.float32)
    )
    num_finalized_batches: jax.Array = dataclasses.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.int32)
    )

    @classmethod
    def empty(cls) -> 'ConcatAndClearArrayMetric.State':
      return cls(
          raw_batch_mean=None,
          accumulated_sum=jnp.zeros((), dtype=jnp.float32),
          num_finalized_batches=jnp.zeros((), dtype=jnp.int32),
      )

    def _consume_raw(self) -> tuple[float, int]:
      if self.raw_batch_mean is None:
        return 0.0, 0
      # Explicitly converts to host float (which fails if called on a JAX tracer
      # or raw ShapeDtypeStruct, just like AutoState and CocoSegmentationAP).
      val = float(self.raw_batch_mean)
      return val, 1

    def merge(
        self, other: 'ConcatAndClearArrayMetric.State'
    ) -> 'ConcatAndClearArrayMetric.State':
      self_val, self_cnt = self._consume_raw()
      other_val, other_cnt = other._consume_raw()
      return dataclasses.replace(
          self,
          raw_batch_mean=None,
          accumulated_sum=(
              jnp.asarray(
                  float(self.accumulated_sum)
                  + float(other.accumulated_sum)
                  + self_val
                  + other_val,
                  dtype=jnp.float32,
              )
          ),
          num_finalized_batches=(
              jnp.asarray(
                  int(self.num_finalized_batches)
                  + int(other.num_finalized_batches)
                  + self_cnt
                  + other_cnt,
                  dtype=jnp.int32,
              )
          ),
      )

    def finalize(self) -> 'ConcatAndClearArrayMetric.State':
      val, cnt = self._consume_raw()
      return dataclasses.replace(
          self,
          raw_batch_mean=None,
          accumulated_sum=jnp.asarray(
              float(self.accumulated_sum) + val, dtype=jnp.float32
          ),
          num_finalized_batches=jnp.asarray(
              int(self.num_finalized_batches) + cnt, dtype=jnp.int32
          ),
      )

    def compute(self) -> dict[str, float]:
      state = self.finalize()
      return {
          'num_batches': float(state.num_finalized_batches),
          'mean_sum': float(state.accumulated_sum),
      }

  def get_state(self, image: Any) -> 'ConcatAndClearArrayMetric.State':
    return self.State(
        raw_batch_mean=jnp.mean(image),
        accumulated_sum=jnp.zeros((), dtype=jnp.float32),
        num_finalized_batches=jnp.zeros((), dtype=jnp.int32),
    )


_TRACE_COUNT = 0
_EXEC_STEPS: list[int] = []


def _reset_counters():
  global _TRACE_COUNT
  _TRACE_COUNT = 0
  _EXEC_STEPS.clear()


class CountingEvaluator(checkpointed_evaluator.CheckpointedEvaluator):
  """An evaluator that counts _step traces and actual device executions."""

  def _step(self, step_nr, state, batch):
    global _TRACE_COUNT
    _TRACE_COUNT += 1
    jax.debug.callback(
        lambda s: _EXEC_STEPS.append(int(s)),
        step_nr,
    )
    return super()._step(step_nr, state, batch)


class StopEvaluator(CountingEvaluator):
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
  _reset_counters()
  with kd.konfig.mock_modules():
    cfg.evals = {
        'stopping_eval': StopEvaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=None,
            metrics={
                'batch_count': CountBatches(0),
                'clear_array': ConcatAndClearArrayMetric(),
            },
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

  # Fresh start should trace _step only once and execute only steps 1 and 2
  # (no wasted throwaway step 0).
  assert _TRACE_COUNT == 1
  assert _EXEC_STEPS == [1, 2]

  # Check that checkpoint for step 2 exists
  assert (ckpt_dir / 'step_0' / 'checkpoints' / 'ckpt_2').exists() or (
      ckpt_dir / '2'
  ).exists()
  assert not (ckpt_dir / 'step_0' / 'checkpoints' / 'ckpt_3').exists()

  # 2. Run evaluation with a CountingEvaluator
  # It should resume from step 2 and execute ONLY steps 3 and 4 (no wasted
  # step 0).
  _reset_counters()
  with kd.konfig.mock_modules():
    cfg.evals = {
        'resuming_eval': CountingEvaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=None,
            metrics={
                'batch_count': CountBatches(0),
                'clear_array': ConcatAndClearArrayMetric(),
            },
            checkpointer=kd.checkpoints.Checkpointer(
                workdir=ckpt_dir, save_interval_steps=1, max_to_keep=None
            ),
        )
    }
  cfg2 = kd.konfig.resolve(cfg)
  evaluator2 = cfg2.evals['resuming_eval']
  merged_aux = evaluator2.evaluate(state, 0)

  assert _EXEC_STEPS == [3, 4]
  assert merged_aux.metric_states['batch_count'].count == 4
  assert int(merged_aux.metric_states['clear_array'].num_finalized_batches) == 4
  assert (ckpt_dir / 'step_0_done').exists()


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
