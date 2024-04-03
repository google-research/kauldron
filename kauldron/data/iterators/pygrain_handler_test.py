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

"""Tests."""

import dataclasses
import pathlib
from typing import Any, SupportsIndex

from grain import python as grain
from kauldron import kd


@dataclasses.dataclass(frozen=True)
class RangeDataSource(grain.RandomAccessDataSource):
  """A data source that yields a range of integers."""

  num_elems: int

  def __getitem__(self, record_key: SupportsIndex) -> Any:
    return int(record_key)

  def __len__(self) -> int:
    return 100


def _make_pipeline() -> kd.data.Pipeline:
  return kd.data.PyGrainPipeline(
      seed=12,
      shuffle=False,
      data_source=RangeDataSource(num_elems=100),
      worker_count=0,
      batch_size=None,
  )


def _make_ckpt(tmp_path: pathlib.Path):
  return kd.ckpts.Checkpointer(
      workdir=tmp_path,
      save_interval_steps=1,
  )


def test_pygrain(tmp_path: pathlib.Path):

  ds_iter = iter(_make_pipeline())
  for _ in range(10):  # Iterate 10 times.
    next(ds_iter)

  ckpt = _make_ckpt(tmp_path)
  ckpt.save(ds_iter, step=1)  # pytype: disable=wrong-arg-types
  ckpt.wait_until_finished()

  # Restoring the pipeline.
  ds_iter = iter(_make_pipeline())
  ckpt = _make_ckpt(tmp_path)
  restored_iter = ckpt.restore(ds_iter)  # pytype: disable=wrong-arg-types
  assert next(restored_iter) == 10
  assert next(restored_iter) == 11
