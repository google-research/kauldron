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

"""ArrayRecord dataset loader."""

import functools
from typing import SupportsIndex
from absl import logging
import grain.python as grain


class LazyArrayRecordDataSource(grain.RandomAccessDataSource):
  """Same as grain.ArrayRecordDataSource but with lazy initialization.

  grain.ArrayRecordDataSource will perform some preprocessing during
  initialization which can be slow. This is especially true when creating a
  kauldron config containing data sources that not be used, such as an eval_ds
  in the config for a training job. This class instead delays creating the
  ArrayRecordDataSource to when the data is first accessed.
  """

  def __init__(self, paths):
    self.paths = paths

  @functools.cached_property
  def _data_source(self):
    logging.info('Initializing ArrayRecordDataSource for paths %s.', self.paths)
    return grain.ArrayRecordDataSource(self.paths)

  def __len__(self) -> int:
    return len(self._data_source)

  def __getitem__(self, record_key: SupportsIndex) -> bytes:
    return self._data_source[record_key]
