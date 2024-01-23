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

"""TF data loader."""

import dataclasses
from typing import Any, Callable, Optional

from grain._src.core import constants
import jax
from kauldron.data.loaders import base
from kauldron.typing import PRNGKey, PRNGKeyLike  # pylint: disable=g-multiple-import,g-importing-member
import tensorflow as tf


@dataclasses.dataclass(repr=False)
class TFData(base.DataLoader):
  """TF data dataset loader."""

  # Function that returns the tf dataset.
  ds_factory: Callable[..., tf.data.Dataset]

  # Post-processing parameters.
  filter_fn: Callable[[Any], bool] | None = None
  # Note: Examples are filtered *before* map_fn is applied.
  map_fn: Callable[[Any], Any] | None = None
  cache: bool = False
  shuffle: bool = True
  shuffle_buffer_size: int = 1_000
  num_epochs: int | None = None

  def __call__(self, seed: Optional[PRNGKeyLike] = None) -> tf.data.Dataset:
    if self.shuffle and seed is None:
      raise ValueError('Shuffling requires a random seed.')

    rngs = 2 * [[None, None]]
    if seed is not None:
      rng = seed if isinstance(seed, PRNGKey) else jax.random.PRNGKey(seed)
      rng = jax.random.fold_in(
          rng, jax.process_index()
      )  # Derive RNG for this host.
      rngs = list(jax.random.split(rng, 2))

    # Load tf.Dataset object
    ds = self.ds_factory()

    if self.filter_fn is not None:
      ds = ds.filter(self.filter_fn)

    if self.map_fn is not None:
      ds = ds.map(self.map_fn)

    if self.cache:
      ds = ds.cache()

    if self.shuffle:
      # Following should be jax.random.bits(rngs.pop()) but this will change the
      # resulting seed.
      ds = ds.shuffle(
          self.shuffle_buffer_size, seed=jax.random.key_data(rngs.pop())[0]
      )
    ds = ds.repeat(self.num_epochs)

    # Add a dummy index (always 0) and a random seed (non-deterministic)
    # to support usage of grain apply_transformations in pipeline.
    def _add_dummy_index_and_non_deterministic_seed(features):
      features[constants.INDEX] = tf.constant((), dtype=tf.int64)
      tf_rng = tf.random.get_global_generator()
      features[constants.SEED] = tf_rng.uniform_full_int((2,), dtype=tf.int32)
      return features

    ds = ds.map(_add_dummy_index_and_non_deterministic_seed)

    return ds
