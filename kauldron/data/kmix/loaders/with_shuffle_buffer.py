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

"""TFDS dataset loader."""

import dataclasses
from typing import ClassVar, Optional, Sequence

from grain._src.tensorflow import transforms as grain_transforms
import grain.tensorflow as grain
import jax
from kauldron import kd
from kauldron import random
from kauldron.data import grain_utils
from kauldron.data.kmix import base
import tensorflow as tf


@dataclasses.dataclass(frozen=True, kw_only=True)
class WithShuffleBuffer(base.TFDataPipeline):
  """Loader with shuffle buffer.

  Note that due to `ds.shuffle`, iterating twice over the `kmix.TFDataPipeline`
  will yield different order. You need to recreate the `kmix.TFDataPipeline`
  object to fully reset the iterator.

  Attributes:
    transforms_before_cache: Data transforms to apply before caching.
    cache: Whether to cache the dataset.
    shuffle: Whether to shuffle the dataset.
    shuffle_buffer_size: Size of the shuffle buffer.
    num_epochs: Number of epochs to repeat the dataset (`None` for infinite
      iteration).
  """

  transforms_before_cache: (
      Sequence[grain.Transformation] | dict[str, grain.Transformation]
  ) = dataclasses.field(default_factory=tuple)

  cache: bool = False
  shuffle: bool = True
  shuffle_buffer_size: Optional[int] = None
  num_epochs: Optional[int] = None

  _supports_symbolic_checkpoint: ClassVar[bool] = False

  def _maybe_apply_pre_cache_transforms(
      self, ds: tf.data.Dataset, *, rng: random.PRNGKey
  ) -> tf.data.Dataset:
    """Applies transforms specified for application before caching."""
    if self.transforms_before_cache:
      ds = grain_utils.maybe_add_grain_meta_features(
          ds,
          rng=rng,
      )
      transforms = []
      if isinstance(self.transforms_before_cache, dict):
        transforms.extend(self.transforms_before_cache.values())
      else:
        transforms.extend(self.transforms_before_cache)
      ds = grain_transforms.apply_transformations(
          ds, self.transforms_before_cache, strict=True
      )
    return ds

  def transform_ds(self, ds, *, rng: kd.random.PRNGKey) -> tf.data.Dataset:
    self._maybe_apply_pre_cache_transforms(ds, rng=rng)

    if self.cache:
      ds = ds.cache()

    if self.shuffle:
      if self.shuffle_buffer_size is None:
        raise ValueError(
            f'`{type(self).__name__}.shuffle_buffer_size` should be specified'
            ' when `shuffle=True`'
        )
      rng = rng.fold_in(jax.process_index())
      rng = rng.fold_in('shuffle_buffer')
      ds = ds.shuffle(
          self.shuffle_buffer_size,
          seed=int(rng.bits()),
          # reshuffle_each_iteration=True,
      )

    ds = ds.repeat(self.num_epochs)
    ds = super().transform_ds(ds, rng=rng)
    return ds
