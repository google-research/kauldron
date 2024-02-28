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

"""Dataset API."""

from __future__ import annotations

import abc
import collections
from collections.abc import Callable
import dataclasses
import functools
import itertools
from typing import Any, Optional

from etils import enp
from etils import epy
from etils import etree
import jax
from kauldron.utils import sharding_utils

_MapFn = Callable[..., Any]
_ArrayIterable = Any


class IterableDataset(abc.ABC):  # Could be generic for better return value
  """General interface for iterable datasets."""

  @property
  @abc.abstractmethod
  def element_spec(self) -> etree.Tree[enp.ArraySpec]:
    """Numpy version of element-spec."""
    raise NotImplementedError()

  @abc.abstractmethod
  def __iter__(self) -> _ArrayIterable:
    raise NotImplementedError()

  def __len__(self) -> int:
    raise TypeError("Unknown length")

  def map(self, map_fn: _MapFn) -> IterableDataset:
    return _MapDataset(parent=self, map_fn=map_fn)

  # Prefetch might improve speed when blocking ops in the
  # CPU (next batch already prefetched), even though it would be best
  # to not have any blocking ops in the host (asynchronous checkpoint saving,
  # metrics, ...).
  def prefetch(self, buffer_size: int = 1):
    """Pre-fetch the iterator (synchronously)."""
    return _PrefetchDataset(parent=self, buffer_size=buffer_size)

  def cache(self):
    return _CachedDataset(parent=self)

  def take(self, num_examples: int) -> IterableDataset:
    return _TakeDataset(parent=self, num_examples=num_examples)

  def device_put(
      self, sharding: Optional[jax.sharding.NamedSharding] = None
  ) -> IterableDataset:
    """Copy elements onto device with a specified sharding.

    Args:
      sharding: How to shard the elements among devices. Will likely be either
        REPLICATED or FIRST_DIM. Defaults to FIRST_DIM.

    Returns:
      A _DevicePutDataset which wraps the original IterableDataset and copies
      all elements onto device.
    """
    if sharding is None:
      sharding = sharding_utils.sharding.FIRST_DIM
    return _DevicePutDataset(parent=self, sharding=sharding)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _DatasetOp(IterableDataset):
  """Base class for implementating operations over iterable datasets."""

  parent: IterableDataset

  @property
  def element_spec(self) -> etree.Tree[enp.ArraySpec]:
    return self.parent.element_spec

  def __iter__(self) -> _ArrayIterable:
    yield from self.parent

  def __len__(self):
    return len(self.parent)

  def __repr__(self):
    return f"{type(self).__name__}({epy.pretty_repr(self.element_spec)})"


@dataclasses.dataclass(frozen=True, kw_only=True)
class _MapDataset(_DatasetOp):
  """Apply the transformation to each elements."""

  map_fn: _MapFn

  def __iter__(self) -> _ArrayIterable:
    for elem in self.parent:
      yield self.map_fn(elem)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PrefetchDataset(_DatasetOp):
  """Pre-fetch the iterator (synchronously)."""

  buffer_size: int

  def __iter__(self) -> _ArrayIterable:
    iterator = iter(self.parent)

    queue = collections.deque()
    # Prefetch buffer size to the queue
    for x in itertools.islice(iterator, self.buffer_size):
      queue.append(x)

    while queue:
      yield queue.popleft()

      # Eventually push the next element to the queue
      try:
        queue.append(next(iterator))
      except StopIteration:
        pass


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TakeDataset(_DatasetOp):
  """Apply the transformation to each elements."""

  num_examples: int

  def __iter__(self) -> _ArrayIterable:
    yield from itertools.islice(self.parent, self.num_examples)

  def __len__(self) -> int:
    return self.num_examples


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CachedDataset(_DatasetOp):
  """Cache the parent iterator."""

  @functools.cached_property
  def cached_examples(self) -> list[Any]:
    try:
      _ = len(self.parent)
    except TypeError as e:
      raise TypeError(
          "Cannot cache an iterator of unknown length. Use take(n) before"
          " cache() to limit the length  of the iterator."
      ) from e
    return list(self.parent)

  def __iter__(self) -> _ArrayIterable:
    yield from self.cached_examples


@dataclasses.dataclass(frozen=True, kw_only=True)
class _DevicePutDataset(_DatasetOp):
  sharding: jax.sharding.NamedSharding

  def __iter__(self) -> _ArrayIterable:
    for elem in self.parent:
      yield sharding_utils.sharding.device_put(elem, self.sharding)
