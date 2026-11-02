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

"""Checkpoint handler tests."""

import typing

import flax
import jax
import jax.numpy as jnp
from kauldron import checkpoints
from kauldron.train import timer as timer_module
import numpy as np


@flax.struct.dataclass
class MyState(checkpoints.items.StandardCheckpointItem):
  x: jax.Array


class _MultiState(typing.NamedTuple):
  state: MyState
  timer: timer_module.PerformanceTimer

  DEFAULT_ITEM = 'state'


class MultiState(_MultiState, checkpoints.items.TopLevelCheckpointItem):
  pass


def test_state(tmp_path):

  state = MyState(x=jnp.ones((1, 2)))

  ckpt = checkpoints.Checkpointer(workdir=tmp_path, save_interval_steps=1)

  assert ckpt.latest_step is None

  ckpt.save(state, step=1)
  ckpt.wait_until_finished()

  new_state = ckpt.restore(MyState(x=jnp.zeros((1, 2))), step=1, donate=False)
  _assert_state(state, new_state)

  new_state = ckpt.restore(checkpoints.items.StandardCheckpointItem(), step=1)
  # The restored structure is a dict
  assert isinstance(new_state, dict)
  np.testing.assert_array_equal(state.x, new_state['x'])


def test_timer(tmp_path):
  timer = timer_module.PerformanceTimer(
      initial_step_num=234,
      initial_training_time_hours=134.0,
      per_device_batch_size=1,
      global_batch_size=1,
  )
  ckpt = checkpoints.Checkpointer(workdir=tmp_path, save_interval_steps=1)
  ckpt.save(timer, step=1)
  ckpt.wait_until_finished()

  timer_to_restore = timer_module.PerformanceTimer(
      initial_step_num=234,
      initial_training_time_hours=44.0,  # different from the saved one
      per_device_batch_size=1,
      global_batch_size=1,
  )
  new_timer = ckpt.restore(timer_to_restore, step=1)
  _assert_timer(timer, new_timer)


def test_multi(tmp_path):
  state = MyState(x=jnp.ones((1, 2)))
  timer = timer_module.PerformanceTimer(
      initial_step_num=234,
      initial_training_time_hours=134.0,
      per_device_batch_size=1,
      global_batch_size=1,
  )

  ckpt = checkpoints.Checkpointer(workdir=tmp_path, save_interval_steps=1)
  ckpt.save(MultiState(state, timer), step=1)
  ckpt.wait_until_finished()

  new_state, new_timer = ckpt.restore(
      MultiState(state, timer), step=1, donate=False
  )
  _assert_state(state, new_state)
  _assert_timer(timer, new_timer)

  # TODO(epot): Test restore only state


def _assert_state(old, new):
  assert isinstance(new, MyState)
  assert old.x.shape == new.x.shape
  np.testing.assert_array_equal(old.x, new.x)


def _assert_timer(old, new):
  assert isinstance(new, timer_module.PerformanceTimer)
  assert old.step_num_when_last_logged == new.step_num_when_last_logged
  assert old.initial_training_time_hours == new.initial_training_time_hours
  assert old.per_device_batch_size == new.per_device_batch_size
  assert old.global_batch_size == new.global_batch_size


def test_cleanup_orphaned_snapshots(tmp_path):
  ckpt_default = checkpoints.Checkpointer(
      workdir=tmp_path, save_interval_steps=1
  )
  assert not ckpt_default.cleanup_orphaned_snapshots
  assert not ckpt_default._ckpt_mgr.cleanup_orphaned_snapshots

  ckpt_enabled = checkpoints.Checkpointer(
      workdir=tmp_path, save_interval_steps=1, cleanup_orphaned_snapshots=True
  )
  assert ckpt_enabled.cleanup_orphaned_snapshots
  assert ckpt_enabled._ckpt_mgr.cleanup_orphaned_snapshots


def test_best_step(tmp_path):
  """Exposes best_step across NoopCheckpointer and Checkpointer modes."""
  assert checkpoints.NoopCheckpointer().best_step is None

  ckpt_no_best = checkpoints.Checkpointer(
      workdir=tmp_path / 'no_best', save_interval_steps=1
  )
  assert ckpt_no_best.best_step is None
  ckpt_no_best.save(MyState(x=jnp.ones((1, 2))), step=10)
  ckpt_no_best.save(MyState(x=jnp.ones((1, 2))), step=20)
  ckpt_no_best.wait_until_finished()
  assert ckpt_no_best.best_step == 20
  ckpt_no_best.close()

  ckpt_best = checkpoints.Checkpointer(
      workdir=tmp_path / 'with_best',
      save_interval_steps=1,
      best_metric_path='eval.loss',
      best_mode='min',
  )
  ckpt_best.save(
      MyState(x=jnp.full((1, 2), 10.0)),
      step=10,
      metrics={'eval': {'loss': 0.5}},
  )
  ckpt_best.save(
      MyState(x=jnp.full((1, 2), 20.0)),
      step=20,
      metrics={'eval': {'loss': 0.2}},
  )
  ckpt_best.save(
      MyState(x=jnp.full((1, 2), 30.0)),
      step=30,
      metrics={'eval': {'loss': 0.4}},
  )
  ckpt_best.wait_until_finished()
  assert ckpt_best.best_step == 20
  assert ckpt_best.latest_step == 30
  ckpt_best.close()


def test_best_metric_keeps_missing_and_evicts_non_finite(tmp_path):
  """Keeps saves without the best metric and evicts NaN losses first."""
  # Saves that lack `best_metric_path` (preemption saves with `{}`, or periodic
  # saves with only `'train'` metrics) are stored with `metrics=None`, so Orbax
  # excludes them from best-N ranking and keeps them. A `NaN` loss ranks worst
  # and is the step evicted once more than `max_to_keep` ranked steps exist:
  #
  #   step | metrics          | ranked as | expected
  #   -----+------------------+-----------+---------
  #   100  | eval.loss = 0.3  | 0.3       | kept (top-2)
  #   200  | eval.loss = 0.2  | 0.2       | kept (best)
  #   250  | {}               | unranked  | kept
  #   275  | train only       | unranked  | kept
  #   300  | eval.loss = NaN  | +inf      | evicted
  ckpt = checkpoints.Checkpointer(
      workdir=tmp_path,
      save_interval_steps=1,
      best_metric_path='eval.loss',
      best_mode='min',
      max_to_keep=2,
  )
  # Finite losses at steps 100 (0.3) and 200 (0.2) occupy the top-2 slots.
  ckpt.save(
      MyState(x=jnp.full((1, 2), 1.0)),
      step=100,
      metrics={'eval': {'loss': 0.3}},
  )
  ckpt.save(
      MyState(x=jnp.full((1, 2), 2.0)),
      step=200,
      metrics={'eval': {'loss': 0.2}},
  )
  # Preemption save with empty metrics is unranked and kept, not a KeyError.
  ckpt.save(
      MyState(x=jnp.full((1, 2), 2.5)),
      step=250,
      force=True,
      metrics={},
  )
  # Periodic save where no evaluator ran: only train metrics, also unranked.
  ckpt.save(
      MyState(x=jnp.full((1, 2), 2.75)),
      step=275,
      metrics={'train': {'loss': 0.1}},
  )
  # Diverged step with NaN loss ranks worst and is evicted.
  ckpt.save(
      MyState(x=jnp.full((1, 2), 3.0)),
      step=300,
      metrics={'eval': {'loss': float('nan')}},
  )
  ckpt.wait_until_finished()
  assert sorted(ckpt.all_steps) == [100, 200, 250, 275]
  assert ckpt.best_step == 200
  ckpt.close()
