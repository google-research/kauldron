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

"""Tfds loader."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any, Optional

from etils import epath
from grain import python as grain
from kauldron.data.py import base
import tensorflow_datasets as tfds


@dataclasses.dataclass(frozen=True)
class DataSource(base.DataSourceBase):
  """Generic loader of arbitrary grain data source."""

  data_source: grain.RandomAccessDataSource


@dataclasses.dataclass(frozen=True)
class Tfds(base.DataSourceBase):
  """Base TFDS loader.

  Convenience wrapper around `tfds.data_source`.
  """

  name: str
  _: dataclasses.KW_ONLY
  split: str
  data_dir: epath.PathLike | None = None
  decoders: Optional[Mapping[str, Any]] = None

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    # TODO(b/373829619): switch back to `tfds.data_source` once that works
    builder = tfds.builder(self.name, data_dir=self.data_dir)
    # if not builder.is_prepared():
    #   builder.download_and_prepare()
    return builder.as_data_source(split=self.split, decoders=self.decoders)
