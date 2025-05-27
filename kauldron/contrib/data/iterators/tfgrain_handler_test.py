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

"""TfGrain handler tests."""

import pathlib

from grain import tensorflow as grain
from kauldron.contrib.data.iterators import tfgrain_handler
from orbax import checkpoint as ocp
import tensorflow as tf


def _make_data_loader() -> grain.TfDataLoader:
  ds = tf.data.Dataset.range(50)
  ds = ds.map(lambda x: x * 10)

  source = grain.TfInMemoryDataSource.from_dataset(ds)
  sampler = grain.TfDefaultIndexSampler(
      num_records=len(source),
      shuffle=False,
      seed=42,
      shard_options=grain.NoSharding(),
  )
  data_loader = grain.TfDataLoader(
      source=source, sampler=sampler, batch_fn=grain.TfBatchNone()
  )
  return data_loader


def test_tfgrain(tmp_path: pathlib.Path):

  ds_iter = iter(_make_data_loader())
  for _ in range(10):  # Iterate 10 times.
    next(ds_iter)

  mgr = ocp.CheckpointManager(tmp_path)
  mgr.save(args=tfgrain_handler.TfGrainArg(ds_iter), step=0)
  mgr.wait_until_finished()

  mgr = ocp.CheckpointManager(tmp_path)
  restored_iter = mgr.restore(
      args=tfgrain_handler.TfGrainArg(iter(_make_data_loader())), step=0
  )

  assert next(restored_iter)['_record'] == 100
  assert next(restored_iter)['_record'] == 110
