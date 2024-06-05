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

"""Pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import dataclasses
import functools
import itertools
from typing import Any, Optional

import jax
from kauldron.data import iterators
from kauldron.data import pipelines
from kauldron.data import utils
import numpy as np

_ArrayTree = Any

# TODO(b/325610230): Delete once PyGrain support efficient batch-lookup


@dataclasses.dataclass(frozen=True, kw_only=True)
class InMemoryPipeline(pipelines.Pipeline):
  """Pipeline which fit in memory.

  Attributes:
    loader: Callable which returns all examples in a single
      `Tree[Array['num_examples ...']] of `np.array`
    shuffle: Whether to shuffle the dataset
    num_epochs: Number of epoch (`None` for infinite iteration)
    drop_remainder: Whether to drop the remainer (currently
      `drop_remainder=False` not supported)
  """

  loader: Callable[[], _ArrayTree]
  shuffle: bool = False
  num_epochs: Optional[int] = None
  drop_remainder: bool = True

  # TODO(epot): Support `transformations=`

  def iter(self) -> Iterator[_ArrayTree]:
    """Iterator."""
    for indices in self.sampler:
      yield jax.tree.map(lambda x: x[indices], self.examples)  # pylint: disable=cell-var-from-loop

  def __iter__(self):
    return iterators.NonCheckpointableIterator(source=self, iter=self.iter())

  def __len__(self):
    return len(self.sampler)

  @functools.cached_property
  def examples(self) -> _ArrayTree:
    """Cached in-memory data."""
    # TODO(epot): Should try to add colab cache across reload.
    return self.loader()

  @functools.cached_property
  def num_examples(self) -> int:
    num_examples_tree = jax.tree.map(lambda x: x.shape[0], self.examples)
    flat_num_examples = jax.tree.leaves(num_examples_tree)
    if not all([shape == flat_num_examples[0] for shape in flat_num_examples]):
      raise ValueError(
          'All features have to have the same number of examples. Got'
          f' {num_examples_tree=}'
      )
    return flat_num_examples[0]

  @property
  def sampler(self) -> BatchedIndexSampler:
    return BatchedIndexSampler(
        batch_size=utils.BatchSize(self.batch_size),
        num_records=self.num_examples,
        num_epochs=self.num_epochs,
        seed=self.seed,
        shuffle=self.shuffle,
        drop_remainder=self.drop_remainder,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class BatchedIndexSampler:
  """Index sampler."""

  batch_size: utils.BatchSize
  num_records: int
  num_epochs: Optional[int]
  seed: Optional[int]
  shuffle: bool
  drop_remainder: bool

  def __iter__(self):
    """Iterator over the examples indices."""
    if self.num_epochs is None:
      iter_epoch = itertools.count()  # Infinite iter
    else:
      iter_epoch = range(self.num_epochs)

    for epoch in iter_epoch:
      all_indices = np.arange(self.num_records)

      if self.shuffle:
        rng = np.random.Philox(key=self.seed)
        rng = rng.jumped(epoch)
        rng = np.random.Generator(rng)
        rng.shuffle(all_indices)

      if self.drop_remainder:
        remainder = len(all_indices) % self.batch_size.total
        if remainder:
          all_indices = all_indices[:-remainder]
      else:
        raise NotImplementedError('drop_remainder is False not supported yet.')

      all_indices = all_indices.reshape((-1, self.batch_size.total))
      # To support multi-host, all host will compute the full indices for all
      # hosts, then each host select a sub-set
      all_indices = all_indices[:, self._start : self._end]
      yield from all_indices

  def __len__(self):
    if self.num_epochs is None:
      raise TypeError('Infinite length.')
    elif self.drop_remainder:
      return (self.num_records // self.batch_size.total) * self.num_epochs
    else:
      raise NotImplementedError('drop_remainder is False not supported yet.')

  @functools.cached_property
  def _start(self) -> int:
    return self.batch_size.per_process * jax.process_index()

  @functools.cached_property
  def _end(self) -> int:
    return self._start + self.batch_size.per_process
