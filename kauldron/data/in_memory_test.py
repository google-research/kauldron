# Copyright 2023 The kauldron Authors.
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

"""Tests."""

from unittest import mock

from kauldron import kd
import numpy as np


def _assert_ds(ds, elems):
  np.testing.assert_array_equal(list(ds), elems)
  assert len(ds) == len(elems)


def test_sampler_no_shuffle():
  ds = kd.data.InMemoryPipeline(
      loader=lambda: np.array([1, 2, 3, 4, 5]),
      batch_size=2,
      num_epochs=1,
  )
  _assert_ds(ds, [[1, 2], [3, 4]])

  ds = kd.data.InMemoryPipeline(
      loader=lambda: np.array([1, 2, 3, 4, 5, 6]),
      batch_size=2,
      num_epochs=1,
  )
  _assert_ds(ds, [[1, 2], [3, 4], [5, 6]])


def test_sampler_shuffle():
  ds = kd.data.InMemoryPipeline(
      loader=lambda: np.array([1, 2, 3, 4, 5]),
      shuffle=True,
      seed=0,
      batch_size=2,
      num_epochs=1,
  )
  _assert_ds(ds, [[5, 2], [3, 1]])

  ds = kd.data.InMemoryPipeline(
      loader=lambda: np.array([1, 2, 3, 4, 5]),
      shuffle=True,
      seed=1,
      batch_size=2,
      num_epochs=1,
  )
  _assert_ds(ds, [[1, 2], [3, 5]])

  ds = kd.data.InMemoryPipeline(
      loader=lambda: np.array([1, 2, 3, 4, 5]),
      shuffle=True,
      seed=1,
      batch_size=2,
      num_epochs=3,
  )
  _assert_ds(
      ds,
      [
          # epoch 0
          [1, 2],
          [3, 5],
          # epoch 1
          [4, 5],
          [1, 2],
          # epoch 2
          [2, 4],
          [3, 1],
      ],
  )


@mock.patch('jax.process_count', return_value=4)
def test_sampler_sharded(_):
  kwargs = dict(
      loader=lambda: np.arange(20),
      shuffle=False,
      batch_size=8,
      num_epochs=1,
  )

  with mock.patch('jax.process_index', return_value=0):
    ds = kd.data.InMemoryPipeline(**kwargs)
    _assert_ds(ds, [[0, 1], [8, 9]])

  with mock.patch('jax.process_index', return_value=1):
    ds = kd.data.InMemoryPipeline(**kwargs)
    _assert_ds(ds, [[2, 3], [10, 11]])

  with mock.patch('jax.process_index', return_value=2):
    ds = kd.data.InMemoryPipeline(**kwargs)
    _assert_ds(ds, [[4, 5], [12, 13]])

  with mock.patch('jax.process_index', return_value=3):
    ds = kd.data.InMemoryPipeline(**kwargs)
    _assert_ds(ds, [[6, 7], [14, 15]])
