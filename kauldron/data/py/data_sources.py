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
import json
from typing import Any, Optional

from etils import epath
from etils import epy
from grain import python as grain
from kauldron.data.py import base
import tensorflow_datasets as tfds

with epy.lazy_imports(
    error_callback=(
        'To use HuggingFace datasets, please install `pip install datasets`.'
    )
):
  import datasets  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


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
    return tfds.data_source(
        self.name,
        split=self.split,
        data_dir=self.data_dir,
        decoders=self.decoders,
    )


@dataclasses.dataclass(frozen=True)
class HuggingFace(base.DataSourceBase):
  """HuggingFace loader."""

  path: str
  config: str | None = None
  _: dataclasses.KW_ONLY
  split: str
  data_dir: epath.PathLike | None = None
  cache_dir: epath.PathLike | None = None

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    return datasets.load_dataset(
        self.path,
        name=self.config,
        split=self.split,
        data_dir=self.data_dir,
        cache_dir=self.cache_dir,
    )


# Should this be part of Grain ?
@dataclasses.dataclass(frozen=True)
class JsonDataSource(grain.RandomAccessDataSource):
  """Json data source.

  Assumes that the json file is a list of examples. The file will be loaded and
  kept in memory.
  """

  path: str

  @functools.cached_property
  def data(self) -> Mapping[str, Any]:
    return json.loads(epath.Path(self.path).read_text())

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, record_key):
    return self.data[record_key]


@dataclasses.dataclass(frozen=True)
class Json(base.DataSourceBase):
  """Json pipeline.

  Assumes that the json file is a list of examples. The file will be loaded and
  kept in memory.
  """

  path: str

  @functools.cached_property
  def data_source(self) -> grain.RandomAccessDataSource:
    return JsonDataSource(path=self.path)
