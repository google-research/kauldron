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

"""Tests for grain_utils."""

import dataclasses
from typing import Any

from grain import tensorflow as grain
from kauldron import kd
from kauldron.data.tf import grain_utils
from kauldron.data.tf import transform_utils
import pytest
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class _MySource(grain.TfDataSource):
  is_supervised: bool = False

  def __len__(self) -> int:
    return 5

  def __getitem__(self, record_keys):
    img = tf.zeros_like(record_keys)
    label = tf.zeros_like(record_keys)

    if self.is_supervised:
      return (img, label)
    else:
      return {'image': img, 'label': label}


@dataclasses.dataclass(frozen=True)
class _MyTransform(kd.data.MapTransform):
  expected_struct: Any

  def map(self, element):
    # Inside map, there's no grain internal keys
    tf.nest.assert_same_structure(self.expected_struct, element)
    return tf.nest.map_structure(lambda x: 12345 + x, element)


@pytest.mark.parametrize('is_supervised', [True, False])
def test_source(is_supervised: bool):
  source = _MySource(is_supervised=is_supervised)
  sampler = grain.TfDefaultIndexSampler(
      num_records=len(source),
      shard_options=grain.NoSharding(),
      num_epochs=1,
  )
  tr = _MyTransform(
      expected_struct=(None, None)
      if is_supervised
      else {'image': None, 'label': None}
  )
  assert not isinstance(tr, grain.MapTransform)
  tr = transform_utils._normalize_transform(tr)
  assert isinstance(tr, grain.MapTransform)
  data_loader = grain.TfDataLoader(
      source=source,
      sampler=sampler,
      transformations=[tr],
  )
  ds = data_loader.as_dataset(start_index=grain.FirstIndex())
  (ex,) = ds.take(1).as_numpy_iterator()

  meta, ex = grain_utils.split_grain_meta_features(ex)

  assert all(k.startswith('_') for k in meta.keys())

  _assert_tree(lambda v: v == 12345, ex)
  _assert_tree(lambda v: v != 12345, meta)


def _assert_tree(fn, ex):
  for v in tf.nest.flatten(ex):
    assert fn(v)
