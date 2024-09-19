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

"""Deprecated symbols."""

# pylint: disable=g-importing-member,unused-import


import abc
import dataclasses
import functools
from typing import Any, Mapping, Optional, TypeAlias

from absl import logging
from etils import edc
from etils import enp
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
from grain._src.tensorflow import transforms
import grain.python as pygrain
import grain.tensorflow as grain
import jax
from kauldron.data import data_utils
from kauldron.data import iterators
from kauldron.data import loaders
from kauldron.data import pipelines
from kauldron.data import utils
from kauldron.data.loaders import base as base_data_loader
from kauldron.typing import PRNGKeyLike, PyTree  # pylint: disable=g-importing-member,g-multiple-import
from kauldron.utils import config_util
import tensorflow as tf
import tensorflow_datasets as tfds


BatchFn = grain.TfBatchFn


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class TFDataPipeline(pipelines.Pipeline):
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
  """

  loader: base_data_loader.DataLoader
  transformations: Any
  # TODO(epot): Users should also be able to specify drop_reminder or mask
  tf_data_options: Optional[tf.data.Options] = None
  prefetch_size: Optional[int] = tf.data.AUTOTUNE

  def __post_init__(self):
    raise ValueError(
        "TFDataPipeline is deprecated."
    )

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
    if self.batch_size:
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
    return ds

  def __iter__(self) -> iterators.Iterator:
    """Iterate over the dataset elements."""
    return iterators.NonCheckpointableIterator(
        source=self, iter=iter(self._ds_iter)
    )

  @property
  def element_spec(self) -> PyTree[enp.ArraySpec]:
    """Returns the element specs of the dataset."""
    return self._ds_iter.element_spec

  def __len__(self) -> int:
    return len(self._ds_iter)


def _drop_grain_meta_features(features: Mapping[str, Any]) -> Mapping[str, Any]:
  return {k: v for k, v in features.items() if k not in grain.META_FEATURES}
