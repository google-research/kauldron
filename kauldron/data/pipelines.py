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

"""Data pipelines."""

import dataclasses
import functools
from typing import Any, Mapping, Optional, TypeAlias

from etils import edc
from etils import enp
from grain._src.tensorflow import transforms
import grain.tensorflow as grain
import jax
from kauldron.data import data_utils
from kauldron.data.loaders import base as base_data_loader
from kauldron.typing import PRNGKeyLike, PyTree  # pylint: disable=g-importing-member,g-multiple-import
from kauldron.utils import config_util
import tensorflow as tf
import tensorflow_datasets as tfds

BatchFn = grain.TfBatchFn

# Output of `tfds.as_numpy`
_NpTfdsDataset: TypeAlias = Any
_NpArray: TypeAlias = Any


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class TFDataPipeline(config_util.UpdateFromRootCfg):
  """Basic tf.data pipeline.

  Attributes:
    loader: A data loader that produces a `tf.data.Dataset`. The dataset should
      be shuffled (if needed) but unbatched.
    transformations: A list of transformations to apply to the dataset. Each
      transformation should be either a `grain.MapTransform` or a
      `grain.RandomMapTransform`.
    batch_size: Global batch size (has to be divisible by number of global
      devices). If specified, then the batch_fn is set to be
      `grain.TfBatch(batch_size=batch_size//num_hosts, drop_remainder=True)`
      Mutually exclusive with `batch_fn` argument.
    batch_fn: A batching function. Note that this transformation has to produce
      batches for individual devices, so it has to use a per-host (not global)
      batch size.
    tf_data_options: An optional tf.data.Options instance to be applied to the
      dataset.
    reshape_for_devices: If True (default) then the per-host batch dimension of
      the dataset elements are reshaped from `(host_batch_size, ...)` to
      `(num_local_devices, host_batch_size/num_local_devices, ...)`.
    prefetch_size: Number of batches to prefetch for this dataset. Defaults to
      AUTOTUNE.
    seed: Optional seed. By default reuse the global seed.
  """

  loader: base_data_loader.DataLoader
  transformations: grain.Transformations
  # TODO(epot): Users should also be able to specify drop_reminder or mask
  batch_size: Optional[int] = None
  tf_data_options: Optional[tf.data.Options] = None
  reshape_for_devices: bool = True
  prefetch_size: Optional[int] = tf.data.AUTOTUNE

  seed: Optional[PRNGKeyLike] = config_util.ROOT_CFG_REF.seed

  @functools.cached_property
  def batch_fn(self) -> BatchFn:
    """Batch transformaton."""
    num_hosts = jax.process_count()
    num_devices = jax.device_count()
    if self.batch_size % num_devices != 0:
      raise ValueError(
          "batch_size must be divisible by num_devices."
          f" {self.batch_size=} {num_devices=}"
      )
    host_batch_size = self.batch_size // num_hosts
    return grain.TfBatch(batch_size=host_batch_size, drop_remainder=True)

  @functools.cached_property
  def _ds_iter(self) -> data_utils.IterableDataset:
    """Returns a numpy tf.data.Dataset iterator."""
    self._assert_root_cfg_resolved()

    ds = self.loader(seed=self.seed)
    transformations = []
    transformations.extend(self.transformations)
    transformations.append(self.batch_fn)
    if self.reshape_for_devices:
      # TODO(epot): Should raise more explicit error message if bash size
      # is not divisible by the number of devices.
      transformations.append(ReshapeForLocalDevices())
    ds = transforms.apply_transformations(ds, transformations, strict=True)

    if self.prefetch_size:
      ds = ds.prefetch(self.prefetch_size)

    # Default data options (can be overwritten using tf_data_options) obtained
    # from https://github.com/google/CommonLoopUtils/tree/HEAD/clu/deterministic_data.py
    ds_options = tf.data.Options()
    ds_options.experimental_optimization.map_parallelization = True
    ds_options.threading.private_threadpool_size = 48
    ds_options.threading.max_intra_op_parallelism = 1
    # Start fetching the data as soon as the `tf.data` pipeline is created
    # (instead of in the first `next(iter(ds))` call) to speed up start time.
    ds_options.experimental_warm_start = True
    if self.tf_data_options is not None:
      ds_options = ds_options.merge(self.tf_data_options)
    ds = ds.with_options(ds_options)

    # drop grain meta features
    ds = ds.map(_drop_grain_meta_features)
    ds = tfds.as_numpy(ds)
    ds = data_utils.IterableDataset(ds)
    return ds

  def __iter__(self) -> PyTree[_NpArray]:
    """Iterate over the dataset elements."""
    return iter(self._ds_iter)

  @property
  def element_spec(self) -> PyTree[enp.ArraySpec]:
    """Returns the element specs of the dataset."""
    return self._ds_iter.element_spec

  __repr__ = edc.repr


def _drop_grain_meta_features(features: Mapping[str, Any]) -> Mapping[str, Any]:
  return {k: v for k, v in features.items() if k not in grain.META_FEATURES}


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class ReshapeForLocalDevices(grain.MapTransform):
  """Reshape elements from [B, ...] -> [N, B // N, ...] with N = jax devices."""

  def map(self, features):
    def _reshape(x):
      n = jax.local_device_count()
      if isinstance(x, tf.Tensor):
        x_shape = tf.shape(x)
        new_shape = tf.concat([[n, x_shape[0] // n], x_shape[1:]], axis=0)
        return tf.reshape(x, new_shape)
      elif x is None:
        return None

    return jax.tree_util.tree_map(_reshape, features)
