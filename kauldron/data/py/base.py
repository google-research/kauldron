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

"""PyGrain data pipeline."""

from __future__ import annotations

import dataclasses
import functools
import math
import typing
from typing import Optional

from etils import epy
import grain.python as grain
import jax
from kauldron import random
from kauldron.data import iterators
from kauldron.data import pipelines
from kauldron.data.py import transform_utils
from kauldron.data.transforms import normalize as tr_normalize
from kauldron.typing import PRNGKeyLike  # pylint: disable=g-importing-member,g-multiple-import


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class PyGrainPipeline(pipelines.Pipeline):
  """Abstract base class to construct PyGrain data pipeline.

  See doc:

  Attributes:
    transforms: A list of transformations to apply to the dataset. Each
      transformation should be either a `grain.MapTransform` or a
      `grain.RandomMapTransform`.
    num_epochs: Number of epoch. If missing, iterate indefinitely (number of
      iteration is given by `cfg.num_training_steps`)
    batch_drop_remainder: Whether or not drop the last examples if `len(ds) %
      batch_size != 0`
    num_workers: how many worker processes to use for data loading (0 to disable
      multiprocessing)
    read_options: Options for reading data from the DataSource.
    enable_profiling: If True data worker process 0 will be profiled.
  """

  # TODO(epot): Need to duplicate `batch_size` because pytype fail to detect
  # the parent class
  if typing.TYPE_CHECKING:
    batch_size: int | None = ...
    seed: PRNGKeyLike | None = ...

  transforms: tr_normalize.Transformations = dataclasses.field(
      default_factory=tuple
  )

  # Params only relevant for the root top-level dataset (when dataset mixture)
  num_epochs: Optional[int] = None
  batch_drop_remainder: bool = True
  num_workers: int = 16
  read_options: grain.ReadOptions | None = None
  enable_profiling: bool = False

  # The pipeline is constructed in 4 functions:
  # * `ds_for_current_process`
  # * `ds_with_transforms`
  # * `_root_map_ds`
  # * `_root_ds`

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    raise NotImplementedError

  def ds_with_transforms(self, rng: random.PRNGKey) -> grain.MapDataset:
    """Create the `tf.data.Dataset` and apply all the transforms."""
    ds = self.ds_for_current_process(rng)

    ds = transform_utils.apply_transforms(ds, self.transforms)

    return ds

  @functools.cached_property
  def _root_map_ds(self) -> grain.MapDataset:
    self._assert_root_cfg_resolved()

    # Loader sometimes uses `jax.random` to generate integer seeds, so allow
    # host<>device here.
    # Instead could force the `PRNGKey` to stay in the CPU.
    with jax.transfer_guard("allow"):
      rng = random.PRNGKey(self.seed)
      rng = rng.fold_in("kmix")
      ds = self.ds_with_transforms(rng)

    # Apply repeat at the end
    ds = ds.repeat(self.num_epochs)
    return ds

  @functools.cached_property
  def _root_ds(self) -> grain.IterDataset:
    # TODO(b/362920968): We're forced to split `_root_ds` into both
    # `_root_map_ds` because `_root_map_ds` does not propagate `len`
    ds = self._root_map_ds
    ds = ds.to_iter_dataset(read_options=self.read_options)
    # We do batching after conversion to `IterDataset` to avoid None during
    # batching.
    if self.batch_size:
      ds = ds.batch(
          self.host_batch_size, drop_remainder=self.batch_drop_remainder
      )

    # Distribute the execution across multiple worker processes.
    num_workers = _get_num_workers(self.num_workers)
    if num_workers > 0:
      multiprocessing_options = grain.MultiprocessingOptions(
          num_workers,
          enable_profiling=self.enable_profiling,
      )
      ds = ds.prefetch(multiprocessing_options)
    return ds

  def __iter__(self) -> iterators.Iterator:
    """Iterate over the dataset elements."""
    return iterators.PyGrainIterator(source=self, iter=iter(self._root_ds))

  def __len__(self) -> int:
    if self.num_epochs is None:
      raise TypeError("Cannot get length of infinite dataset.")

    ds_len = len(self._root_map_ds)
    if not self.batch_size:
      return ds_len

    if self.batch_drop_remainder:
      return ds_len // self.batch_size
    else:
      return math.ceil(ds_len / self.batch_size)

  def __getitem__(self, record_key: int):
    """Get an item from the dataset."""
    ds = self._root_map_ds
    # Re-apply the batching, as in the standard pipeline, it is applied after
    # conversion to `IterDataset`.
    if self.batch_size:
      ds = ds.batch(
          self.host_batch_size, drop_remainder=self.batch_drop_remainder
      )
    return ds[record_key]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class DataSourceBase(PyGrainPipeline):
  """Base class to implement a data source.

  Child classes should overwrite the `data_source` property.
  See `kd.data.py.Tfds` for an example.

  Attributes:
    shuffle: whether to shuffle
  """

  shuffle: bool
  data_source: grain.RandomAccessDataSource = dataclasses.field(
      init=False,
      repr=False,
  )

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    ds = grain.MapDataset.source(self.data_source)
    ds = ds.seed(rng.as_seed())

    # Shard the dataset
    ds = ds[jax.process_index() :: jax.process_count()]

    # Global shuffle
    if self.shuffle:
      # Here we explicitly pass the seed.
      # This allow the example order to be constant even if transformations
      # changes. Note that this might be the case anyway but I haven't looked
      # at grain's internal and how seed is derived to the individual
      # transformation.
      ds = ds.shuffle(seed=rng.fold_in("shuffle").as_seed())
    return ds


def _get_num_workers(num_workers: int) -> int:
  """Set the number of workers."""
  if epy.is_notebook() or epy.is_test():  # in colab worker_count has to be 0
    # TODO(klausg): autodetect if Kernel supports multiprocessing
    # Could check
    # from multiprocessing import spawn
    # spawn.get_executable() is not None
    print("Disabling pygrain multi-processing (unsupported in colab).")
    return 0
  else:
    return num_workers
