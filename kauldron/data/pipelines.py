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
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
from grain._src.tensorflow import transforms
import grain.python as pygrain
import grain.tensorflow as grain
import jax
from kauldron.data import data_utils
from kauldron.data.loaders import base as base_data_loader
from kauldron.typing import PRNGKeyLike, PyTree  # pylint: disable=g-importing-member,g-multiple-import
from kauldron.utils import config_util
from kauldron.utils.sharding_utils import sharding  # pylint: disable=g-importing-member
import tensorflow as tf
import tensorflow_datasets as tfds

BatchFn = grain.TfBatchFn

# Output of `tfds.as_numpy`
_NpTfdsDataset: TypeAlias = Any
_NpArray: TypeAlias = Any


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class Pipeline(config_util.UpdateFromRootCfg):
  """Base class for kauldron data pipelines.

  Attributes:
    batch_size: Global batch size. Has to be divisible by number of global
      devices. Pipeline should take care of sharding the data between hosts.
    seed: Random seed to be used for things like shuffling and randomness in
      preprocessing. Defaults to the seed from the root config.
  """

  batch_size: int
  seed: Optional[PRNGKeyLike] = config_util.ROOT_CFG_REF.seed

  def __iter__(self) -> PyTree[_NpArray]:
    """Iterator that produces batches as PyTrees of Sharded jax.Arrays."""
    raise NotImplementedError()

  @functools.cached_property
  def element_spec(self) -> PyTree[enp.ArraySpec]:
    """Returns the element specs of a single batch."""
    first_elem = next(iter(self))
    return etree.spec_like(first_elem)

  @functools.cached_property
  def host_batch_size(self) -> int:
    num_hosts = jax.process_count()
    num_devices = jax.device_count()
    if self.batch_size % num_devices != 0:
      raise ValueError(
          "batch_size must be divisible by num_devices."
          f" {self.batch_size=} {num_devices=}"
      )
    return self.batch_size // num_hosts

  __repr__ = edc.repr


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class TFDataPipeline(Pipeline):
  """Basic tf.data pipeline.

  Attributes:
    loader: A data loader that produces a `tf.data.Dataset`. The dataset should
      be shuffled (if needed) but unbatched.
    transformations: A list of transformations to apply to the dataset. Each
      transformation should be either a `grain.MapTransform` or a
      `grain.RandomMapTransform`.
    tf_data_options: An optional tf.data.Options instance to be applied to the
      dataset.
    prefetch_size: Number of batches to prefetch for this dataset. Defaults to
      AUTOTUNE.
    seed: Optional seed. By default reuse the global seed.
  """

  loader: base_data_loader.DataLoader
  transformations: grain.Transformations
  # TODO(epot): Users should also be able to specify drop_reminder or mask
  tf_data_options: Optional[tf.data.Options] = None
  prefetch_size: Optional[int] = tf.data.AUTOTUNE

  @functools.cached_property
  def batch_fn(self) -> BatchFn:
    """Batch transformaton."""
    return grain.TfBatch(batch_size=self.host_batch_size, drop_remainder=True)

  @functools.cached_property
  def _ds_iter(self) -> data_utils.IterableDataset:
    """Returns a numpy tf.data.Dataset iterator."""
    self._assert_root_cfg_resolved()

    # Loader sometimes uses `jax.random` to generate integer seeds, so allow
    # host<>device here.
    with jax.transfer_guard("allow"):
      ds = self.loader(seed=self.seed)
    transformations = []
    transformations.extend(self.transformations)
    transformations.append(self.batch_fn)
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
    # Shard the batch across the available devices
    ds = ds.map(lambda ex: sharding.device_put(ex, sharding.SHARDED))
    return ds

  def __iter__(self) -> PyTree[_NpArray]:
    """Iterate over the dataset elements."""
    return iter(self._ds_iter)

  @property
  def element_spec(self) -> PyTree[enp.ArraySpec]:
    """Returns the element specs of the dataset."""
    return self._ds_iter.element_spec


def _drop_grain_meta_features(features: Mapping[str, Any]) -> Mapping[str, Any]:
  return {k: v for k, v in features.items() if k not in grain.META_FEATURES}


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class PyGrainPipeline(Pipeline):
  """Basic pygrain pipeline.

  Attributes:
    data_source: a random access datasource
    transformations: A list of transformations to apply to the dataset. Each
      transformation should be either a `grain.MapTransform` or a
      `grain.RandomMapTransform`.
    shuffle: whether to shuffle
    worker_count: how many worker processes to use for data loading
  """

  data_source: pygrain.RandomAccessDataSource
  transformations: grain.Transformations
  shuffle: bool
  num_epochs: Optional[int] = None
  worker_count: int = 16

  @functools.cached_property
  def batch_fn(self) -> pygrain.Batch:
    """Batch transformaton."""
    # TODO(klausg): Users should also be able to specify drop_reminder or mask
    return pygrain.Batch(self.host_batch_size, drop_remainder=True)

  @property
  def sampler(self) -> pygrain.Sampler:
    return pygrain.IndexSampler(
        num_records=len(self.data_source),
        num_epochs=None,
        seed=self.seed,
        shuffle=self.shuffle,
        shard_options=pygrain.ShardByJaxProcess(),
    )

  @functools.cached_property
  def _ds_iter(self) -> data_utils.IterableDataset:
    """Returns a numpy tf.data.Dataset iterator."""
    self._assert_root_cfg_resolved()

    transformations = []
    transformations.extend(self.transformations)
    transformations.append(self.batch_fn)

    transformations.append(DevicePut())

    worker_count = self.worker_count
    if epy.is_notebook():  # in colab worker_count has to be 0
      # TODO(klausg): autodetect if Kernel supports multiprocessing
      print("Disabling pygrain multi-processing (unsupported in colab).")
      worker_count = 0

    dataloader = pygrain.DataLoader(
        data_source=self.data_source,
        operations=transformations,
        sampler=self.sampler,
        worker_count=worker_count,
        shard_options=pygrain.ShardByJaxProcess(),
    )
    return data_utils.IterableDataset(dataloader)

  def __iter__(self) -> PyTree[_NpArray]:
    """Iterate over the dataset elements."""
    return iter(self._ds_iter)


class DevicePut(pygrain.MapTransform):
  """Put the batch onto device in a sharded way."""

  def map(self, batch):
    return sharding.device_put(batch, sharding.SHARDED)
