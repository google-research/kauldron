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

"""PyGrain data pipeline."""

from __future__ import annotations

import dataclasses
import enum
import functools
import math
import os
import typing
from typing import Callable
from typing import Optional
from typing import TypeVar

from etils import enp
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import grain.python as grain
import jax
from kauldron import random
from kauldron.data import iterators
from kauldron.data import pipelines
from kauldron.data.py import transform_utils
from kauldron.data.transforms import normalize as tr_normalize
from kauldron.ktyping import PyTree  # pylint: disable=g-importing-member
from kauldron.typing import PRNGKeyLike  # pylint: disable=g-importing-member
import tensorflow as tf

_ConsistentDatasetType = TypeVar(
    "_ConsistentDatasetType", grain.MapDataset, grain.IterDataset
)


class DropRemainder(enum.StrEnum):
  """Determines how to handle a final partial batch of a dataset.

  Attributes:
    DROP: Drop the last partial batch.
    KEEP: Keep the last partial batch as is (may have size < `batch_size`).
    PAD: Pad the last partial batch with zeros to reach `batch_size`.
  """

  DROP = enum.auto()
  KEEP = enum.auto()
  PAD = enum.auto()


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
    batch_drop_remainder: What to do with the last partial batch. See
      `DropRemainder` for possible values. True is equivalent to
      `DropRemainder.DROP`, False is equivalent to `DropRemainder.KEEP`.
    num_workers: how many worker processes to use for data loading (0 to disable
      multiprocessing)
    read_options: Options for reading data from the DataSource.
    enable_profiling: If True data worker process 0 will be profiled.
    shard_by_process: Whether to shard the dataset by process count and index.
      Should use the default for most cases. Use non-default values for
      experimental use only.
    worker_init_fn: If set, will initialize subprocesses with this function
      instead of the kauldron default.
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
  batch_drop_remainder: bool | str | DropRemainder = True
  num_workers: int = 16
  read_options: grain.ReadOptions | None = None
  enable_profiling: bool = False
  per_worker_buffer_size: int = 1
  shard_by_process: bool = True

  worker_init_fn: Callable[[int, int], None] | None = None

  def __post_init__(self):
    if hasattr(super(), "__post_init__"):
      super().__post_init__()  # Future proof to run `__post_init__` in parents  # pylint: disable=attribute-error

    object.__setattr__(
        self,
        "batch_drop_remainder",
        _normalize_batch_drop_remainder(self.batch_drop_remainder),
    )

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

  def _apply_batch(self, ds: _ConsistentDatasetType) -> _ConsistentDatasetType:
    if not isinstance(self.batch_drop_remainder, DropRemainder):
      raise TypeError("Did you forgot to call super().__post_init__ ?")

    match self.batch_drop_remainder:
      case DropRemainder.PAD:
        batch_fn = functools.partial(
            grain.experimental.batch_and_pad, batch_size=self.host_batch_size
        )
        drop_remainder = False
      case DropRemainder.KEEP:
        batch_fn = None
        drop_remainder = False
      case DropRemainder.DROP:
        batch_fn = None
        drop_remainder = True
      case _:
        # Unreachable, but leaving it here to protect against future additions
        # to the enum.
        raise ValueError("Invalid batch_drop_remainder: {batch_drop_remainder}")
    return ds.batch(
        self.host_batch_size, batch_fn=batch_fn, drop_remainder=drop_remainder
    )

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

  def _make_root_ds(self, *, num_workers: int) -> grain.IterDataset:
    # TODO(b/362920968): We're forced to split `_root_ds` into both
    # `_root_map_ds` because `_root_map_ds` does not propagate `len`
    ds = self._root_map_ds

    if num_workers == 0 or self.read_options is None:
      # TODO(epot): Fix adhoc import thread-safety and restore this !!!
      # PyGrain multi-thread has various issues, so disable it:
      # * einops backend registration is not thread-safe.
      # * Adhoc imports are not thread-safe, creating issues when lazy-imports
      #   get triggered inside the data pipeline.
      # So disable pre-fetching added by `to_iter_dataset`.
      read_options = grain.ReadOptions(
          num_threads=0,
          prefetch_buffer_size=0,
      )
    else:
      read_options = self.read_options

    # a prefetch at the end. When.
    ds = ds.to_iter_dataset(read_options=read_options)
    # We do batching after conversion to `IterDataset` to avoid None during
    # batching.
    if self.batch_size:
      ds = self._apply_batch(ds)

    # Distribute the execution across multiple worker processes.
    if num_workers > 0:
      multiprocessing_options = grain.MultiprocessingOptions(
          num_workers=num_workers,
          enable_profiling=self.enable_profiling,
          per_worker_buffer_size=self.per_worker_buffer_size,
      )
      ds = ds.mp_prefetch(
          multiprocessing_options,
          worker_init_fn=functools.partial(
              _worker_init_fn, custom_worker_init_fn=self.worker_init_fn
          ),
      )
    return ds

  @functools.cached_property
  def _root_ds(self) -> grain.IterDataset:
    # Distribute the execution across multiple worker processes.
    num_workers = _get_num_workers(self.num_workers)

    return self._make_root_ds(num_workers=num_workers)

  @functools.cached_property
  def element_spec(self) -> PyTree[enp.ArraySpec]:
    """Returns the element specs of a single batch."""
    # To avoid the memory overhead of multiprocessing when `num_workers` is
    # large, explicitly turn off multiprocessing here since we only require
    # the first element.
    ds = self._make_root_ds(num_workers=0)
    first_elem = next(iter(ds))
    return etree.spec_like(first_elem)

  def __iter__(self) -> iterators.Iterator:
    """Iterate over the dataset elements."""
    return iterators.PyGrainIterator(source=self, iter=iter(self._root_ds))

  def __len__(self) -> int:
    if self.num_epochs is None:
      raise TypeError("Cannot get length of infinite dataset.")

    ds_len = len(self._root_map_ds)
    if not self.batch_size:
      return ds_len

    if self.batch_drop_remainder == DropRemainder.DROP:
      return ds_len // self.host_batch_size
    else:
      return math.ceil(ds_len / self.host_batch_size)

  def __getitem__(self, record_key: int):
    """Get an item from the dataset."""
    ds = self._root_map_ds
    # Re-apply the batching, as in the standard pipeline, it is applied after
    # conversion to `IterDataset`.
    if self.batch_size:
      ds = self._apply_batch(ds)
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
    if self.shard_by_process:
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


def _worker_init_fn(
    worker_idx: int,
    worker_count: int,
    *,
    custom_worker_init_fn: Callable[[int, int], None] | None = None,
):
  """Prevent excessive GPU memory allocation in the dataloader workers.

  Prevents TensorFlow from using the GPU and jax preallocation of GPU memory in
  the data reader workers.

  Args:
    worker_idx: Unused index of the worker.
    worker_count: Unused number of workers.
    custom_worker_init_fn: optional function to initialize worker.
  """
  # Prevent TensorFlow from using the GPU.
  tf.config.set_visible_devices([], "GPU")

  # Prevent jax from preallocating GPU memory in the data reader workers.
  # See https://docs.jax.dev/en/latest/gpu_memory_allocation.html
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

  if custom_worker_init_fn is not None:
    custom_worker_init_fn(worker_idx, worker_count)


def _normalize_batch_drop_remainder(
    batch_drop_remainder: bool | str | DropRemainder,
) -> DropRemainder:
  """Normalize the batch drop remainder to an enum value."""
  if batch_drop_remainder is True:  # pylint: disable=g-bool-id-comparison
    return DropRemainder.DROP
  elif batch_drop_remainder is False:  # pylint: disable=g-bool-id-comparison
    return DropRemainder.KEEP
  else:
    return DropRemainder(batch_drop_remainder)
