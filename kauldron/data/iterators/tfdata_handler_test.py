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

"""Test."""

import pathlib

from kauldron.data.iterators import tfdata_handler
from orbax import checkpoint as ocp
import tensorflow as tf


def test_orbax(tmp_path: pathlib.Path):
  ds = tf.data.Dataset.range(50)
  ds = ds.map(lambda x: x * 10)

  ds_iter = iter(ds)
  for _ in range(10):  # Iterate 10 times.
    next(ds_iter)

  mgr = ocp.CheckpointManager(tmp_path)
  mgr.save(args=tfdata_handler.TFDataArg(ds_iter), step=0)

  mgr = ocp.CheckpointManager(tmp_path)
  restored_iter = mgr.restore(args=tfdata_handler.TFDataArg(iter(ds)), step=0)

  assert next(restored_iter).numpy() == 100
  assert next(restored_iter).numpy() == 110
