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
  np.testing.assert_array_equal(state.x, new_state['x'])  # pytype: disable=unsupported-operands


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
