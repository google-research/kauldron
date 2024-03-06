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

"""Iterator."""

from __future__ import annotations

import abc
import dataclasses
from typing import TypeVar

from etils import enp
from etils import etree
from kauldron import checkpoints
from kauldron.data import data_utils


_FnT = TypeVar('_FnT')


@dataclasses.dataclass(frozen=True, kw_only=True)
class Iterator(checkpoints.items.CheckpointItem):  # pytype: disable=ignored-abstractmethod
  """Wrapper around a dataset iterator.

  Adds:

  * Checkpoint support
  * `len` and `.element_spec` support
  """
  source: data_utils.IterableDataset

  def __iter__(self):
    return self

  @abc.abstractmethod
  def __next__(self):
    raise NotImplementedError

  def __len__(self):
    return len(self.source)

  @property
  def element_spec(self) -> etree.Tree[enp.ArraySpec]:
    """Numpy version of element-spec."""
    return self.source.element_spec
