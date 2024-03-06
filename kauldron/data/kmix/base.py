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

"""Data pipelines."""

import abc
from collections.abc import Sequence
import dataclasses
import functools
import sys
from typing import Any, Iterator, Optional, TypeAlias

from etils import enp
from grain._src.tensorflow import transforms as grain_transforms
import grain.tensorflow as grain
import jax
from kauldron import random
from kauldron.data import grain_utils
from kauldron.data import pipelines
from kauldron.typing import PyTree  # pylint: disable=g-importing-member,g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds


# Output of `tfds.as_numpy`
_NpTfdsDataset: TypeAlias = Any
_NpArray: TypeAlias = Any

# Sentinel value that indicate the field should only be activated at the top
# level node.
_ROOT_ONLY: Any = object()


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class Base(pipelines.Pipeline, abc.ABC):
  """Base classes for all `tf.data` pipeline.

  Subclasses should only implement the `ds_for_current_process` method which
  returns the `tf.data.Dataset` for the current process.

  Attributes:
    batch_size: Batch size.
    transforms: A list of `grain.Transformation` to apply to the dataset. Can be
      a dict to allow easier CLI / sweep access (
      `--cfg.train_ds.transforms.img_scale.in_vrange=(-1,1)`)
    tf_data_options: An optional tf.data.Options instance to be applied to the
      dataset.
    prefetch_size: Number of batches to prefetch for this dataset. Defaults to
      AUTOTUNE.
  """

  # TODO(epot): Users should also be able to specify drop_reminder or mask
  batch_size: int | None = None
  batch_drop_remainder: bool = True
  transforms: (
      Sequence[grain.Transformation] | dict[str, grain.Transformation]
  ) = dataclasses.field(default_factory=tuple)

  # Those fields are only applied once at the top level
  tf_data_options: Optional[tf.data.Options] = None
  prefetch_size: Optional[int] = _ROOT_ONLY

  # ======================== Protocol ========================

  @abc.abstractmethod
  def ds_for_current_process(self, rng: random.PRNGKey) -> tf.data.Dataset:
    """Returns the dataset for the current process.

    Args:
      rng: A PRNGKey used to sample the dataset. The `rng` is the same for all
        processes, so it's the users responsibility to call
        `rng.fold_in(jax.process_index())` if required.

    Returns:
      The `tf.data.Dataset` for the current process. Each process should yield
        non-overlapping examples.
    """
    raise NotImplementedError

  def transform_ds(
      self, ds: tf.data.Dataset, *, rng: random.PRNGKey
  ) -> tf.data.Dataset:
    """Eventually apply transforms to the dataset."""
    ds = _maybe_add_grain_meta_features(
        ds,
        rng=rng,
    )

    ds = self._apply_transforms(ds, self.batch_size)
    return ds

  # ======================== Public API ========================

  def __iter__(self) -> Iterator[PyTree[_NpArray]]:
    """Iterate over the dataset elements."""
    return iter(self._root_ds_iter)

  @property
  def element_spec(self) -> PyTree[enp.ArraySpec]:
    """Returns the element specs of the dataset."""
    return self._root_ds_iter.element_spec

  def __len__(self) -> int:
    return len(self._root_ds_iter)

  # ======================== Internals ========================

  @functools.cached_property
  def _root_ds_iter(self) -> _NpTfdsDataset:
    """Returns a numpy tf.data.Dataset iterator."""
    self._assert_root_cfg_resolved()

    # Loader sometimes uses `jax.random` to generate integer seeds, so allow
    # host<>device here.
    # Instead could force the `PRNGKey` to stay in the CPU.
    with jax.transfer_guard("allow"):
      rng = random.PRNGKey(self.seed)
      rng = rng.fold_in("kmix")
      ds = self.ds_with_transforms(rng, _is_root=True)

    # Drop grain meta features
    ds = ds.map(lambda ex: grain_utils.split_grain_meta_features(ex)[1])

    ds = tfds.as_numpy(ds)
    return ds

  def ds_with_transforms(
      self, rng: random.PRNGKey, *, _is_root: bool = False
  ) -> tf.data.Dataset:
    """Create the `tf.data.Dataset` and apply all the transforms."""
    ds = self.ds_for_current_process(rng)
    ds = self.transform_ds(ds, rng=rng)

    # Additional transformations applied only at the top level
    ds = self._apply_options(ds, is_root=_is_root)

    # Only apply `prefetch` at the end of the pipeline.
    prefetch_size = _get_if_root(
        self.prefetch_size,
        default=tf.data.AUTOTUNE,
        is_root=_is_root,
    )
    if prefetch_size:
      ds = ds.prefetch(prefetch_size)

    return ds

  def _apply_transforms(
      self, ds: tf.data.Dataset, batch_size: int | None
  ) -> tf.data.Dataset:
    """Apply transforms to the dataset."""

    transforms = []
    if isinstance(self.transforms, dict):
      transforms.extend(self.transforms.values())
    else:
      transforms.extend(self.transforms)
    if batch_size:
      transforms.append(
          grain.TfBatch(
              batch_size=self.host_batch_size,
              drop_remainder=self.batch_drop_remainder,
          )
      )
    ds = grain_transforms.apply_transformations(ds, transforms, strict=True)
    return ds

  def _apply_options(
      self, ds: tf.data.Dataset, *, is_root: bool = False
  ) -> tf.data.Dataset:
    """Apply `tf.data.Options` to the dataset."""
    if is_root:
      # Default data options (can be overwritten using tf_data_options) obtained
      # from https://github.com/google/CommonLoopUtils/tree/HEAD/clu/deterministic_data.py
      ds_options = tf.data.Options()
      ds_options.experimental_optimization.map_parallelization = True
      ds_options.threading.private_threadpool_size = 48
      ds_options.threading.max_intra_op_parallelism = 1
      # Start fetching the data as soon as the `tf.data` pipeline is created
      # (instead of in the first `next(iter(ds))` call) to speed up start time.
      ds_options.experimental_warm_start = True
    else:
      ds_options = None

    if self.tf_data_options is not None:
      if is_root:
        assert ds_options is not None
        ds_options = ds_options.merge(self.tf_data_options)
      else:
        ds_options = self.tf_data_options

    if ds_options is not None:
      ds = ds.with_options(ds_options)
    return ds


def _maybe_add_grain_meta_features(
    ds,
    *,
    rng: random.PRNGKey,
) -> Any:
  """Add grain meta features."""
  # Dataset already has grain meta features.
  if isinstance(ds.element_spec, dict) and grain.INDEX in ds.element_spec:
    return ds

  # This should be deterministic as long as the `ds` is deterministic.
  sampler = grain.TfDefaultIndexSampler(
      num_records=sys.maxsize,  # Infinite iterator
      shuffle=False,
      seed=int(rng.fold_in("grain_metadata").bits()),
      shard_options=grain.ShardByJaxProcess(),
      num_epochs=None,
  )
  index_ds = sampler.get_index_dataset(grain.FirstIndex())
  ds = tf.data.Dataset.zip(index_ds, ds)
  ds = ds.map(grain_utils.merge_grain_meta_features)
  return ds


def _get_if_root(value, *, default, is_root: bool):
  if value is _ROOT_ONLY:
    if is_root:
      return default
    else:
      return None
  else:
    return value
