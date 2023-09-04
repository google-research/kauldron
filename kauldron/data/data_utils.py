# Copyright 2023 The kauldron Authors.
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

"""Dataset API."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any

from etils import enp
from etils import epy
from etils import etree

_MapFn = Callable[..., Any]
_ArrayIterable = Any


# TODO(epot): This could go inside TFDS directly
# TODO(epot): Should this be merged with `kd.data.TFDataPipeline` ? So user
# can add additional transformations in Colab ?
@dataclasses.dataclass(frozen=True)
class IterableDataset:  # Could be generic for better return value
  """`tf.data`-like API for `np.array` / `jax.Array`."""

  parent: IterableDataset

  def map(self, map_fn: _MapFn) -> IterableDataset:
    return _MapDataset(self.parent, map_fn=map_fn)

  # TODO(epot): Add a prefetch might improve speed when blocking ops in the
  # CPU (next batch already prefetched), even though it would be best
  # to not have any blocking ops in the host (asynchronous checkpoint saving,
  # metrics, ...).
  # def prefetch(self, i):
  #   return _PrefetchDataset()

  @property
  def element_spec(self) -> etree.Tree[enp.ArraySpec]:
    """Numpy version of element-spec."""
    return self.parent.element_spec

  def __iter__(self) -> _ArrayIterable:
    for elem in self.parent:
      yield elem

  def __len__(self) -> int:
    return len(self.parent)

  def __repr__(self):
    return f'{type(self).__name__}({epy.pretty_repr(self.element_spec)})'


@dataclasses.dataclass(frozen=True)
class _MapDataset(IterableDataset):
  """Apply the transformation to each elements."""

  _: dataclasses.KW_ONLY
  map_fn: _MapFn

  def __iter__(self) -> _ArrayIterable:
    for elem in self.parent:
      yield self.map_fn(elem)
