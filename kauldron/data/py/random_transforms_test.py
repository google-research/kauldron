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

from absl.testing import absltest
from absl.testing import parameterized
from grain import python as grain
from kauldron.data.py import random_transforms
import numpy as np


class _DictDataSource(grain.RandomAccessDataSource):
  """Returns dicts with arange arrays at each index."""

  def __init__(self, num_elements: int):
    self._num_elements = num_elements

  def __len__(self) -> int:
    return self._num_elements

  def __getitem__(self, index):
    return {"image": np.arange(8 * 8 * 3, dtype=np.float32).reshape(8, 8, 3)}


class ElementWiseRandomTransformDeterminismTest(parameterized.TestCase):

  def test_repeated_access_same_index_returns_same_value(self):
    """ds[i] read twice from the same pipeline must return identical values."""
    ds = grain.MapDataset.source(_DictDataSource(num_elements=4))
    ds = ds.seed(42)
    ds = ds.random_map(
        random_transforms.RandomCrop(key="image", shape=(4, 4, 3))
    )

    first_read = ds[0]["image"].copy()
    second_read = ds[0]["image"].copy()

    np.testing.assert_array_equal(first_read, second_read)

  @parameterized.parameters(1, 2)
  def test_two_pipelines_same_seed_return_same_values(self, num_iters):
    """Two pipelines built with the same seed must produce identical output."""
    ds1 = grain.MapDataset.source(_DictDataSource(num_elements=4))
    ds1 = ds1.seed(42)
    ds1 = ds1.random_map(
        random_transforms.RandomCrop(key="image", shape=(4, 4, 3))
    )
    ds1_iter = iter(ds1)

    ds2 = grain.MapDataset.source(_DictDataSource(num_elements=4))
    ds2 = ds2.seed(42)
    ds2 = ds2.random_map(
        random_transforms.RandomCrop(key="image", shape=(4, 4, 3))
    )
    ds2_iter = iter(ds2)

    for _ in range(num_iters):
      result1 = next(ds1_iter)
    for _ in range(num_iters):
      result2 = next(ds2_iter)

    np.testing.assert_array_equal(result1["image"], result2["image"])


if __name__ == "__main__":
  absltest.main()
