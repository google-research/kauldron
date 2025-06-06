# Copyright 2025 The kauldron Authors.
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

"""Tests the Millstone iterator."""

import dataclasses
import functools
from typing import Any

from absl.testing import absltest
import grain.python as grain
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.contrib.millstone import millstone_iterator
import numpy as np
from orbax import checkpoint as ocp
import tensorflow as tf


class MillstoneDataset(kd.data.IterableDataset):
  """A test Millstone dataset."""

  def __init__(self, batch_size: int):
    self._batch_size = batch_size
    self._remote_dataset = self._make_remote_dataset(batch_size)

  def _make_remote_dataset(
      self, batch_size: int
  ) -> remote_dataset.RemoteDataset:

    def _make_iterator(
        shard_index: int, shard_count: int, batch_size: int
    ) -> remote_dataset.CheckpointableIterator:
      dataset = grain.MapDataset.range(100)
      dataset = dataset[shard_index::shard_count]
      dataset = dataset.batch(batch_size, drop_remainder=True)
      dataset = dataset.to_iter_dataset()
      return dataset.__iter__()

    return remote_dataset.RemoteDataset(
        lambda shard_index, shard_count: _make_iterator(
            shard_index, shard_count, batch_size
        ),
        element_spec=self.element_spec,
    )

  def __iter__(self) -> millstone_iterator.MillstoneIterator:
    return millstone_iterator.MillstoneIterator(
        source=self, iter=self._remote_dataset.__iter__()
    )

  @functools.cached_property
  def element_spec(self) -> Any:
    return {"x": jax.ShapeDtypeStruct((self._batch_size,), jnp.int32)}


class MillstoneIteratorTest(absltest.TestCase):
  """Tests the Millstone iterator."""

  def test_millstone_iterator(self):

    tmp_path = self.create_tempdir()
    dataset = MillstoneDataset(batch_size=4)
    it = dataset.__iter__()
    for _ in range(2):
      next(it)

    # Saves the checkpoint.
    ckpt = kd.ckpts.Checkpointer(workdir=tmp_path, save_interval_steps=1)
    ckpt.save(it, step=1)
    ckpt.wait_until_finished()
    expected = [next(it) for _ in range(2)]

    # Restores from the checkpoint.
    it = dataset.__iter__()
    ckpt = kd.ckpts.Checkpointer(workdir=tmp_path, save_interval_steps=1)
    restored_iter = ckpt.restore(it)
    data = [next(restored_iter) for _ in range(2)]

    np.testing.assert_equal(data, expected)


if __name__ == "__main__":
  jax.config.parse_flags_with_absl()
  googletest_launcher.main()
