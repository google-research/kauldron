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
from typing import Any, Callable, Mapping, Optional

from absl import logging
from grain._src.core import constants
import jax
from kauldron.data.loaders import base
from kauldron.typing import PRNGKey, PRNGKeyLike  # pylint: disable=g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass(repr=False)
class Tfds(base.DataLoader):
  """Basic TFDS dataset loader."""

  # tfds.builder parameters.
  name: str
  data_dir: Optional[str] = None
  split: str = "train"
  decoders: Optional[Mapping[str, Any]] = None

  # Post-processing parameters.
  data_limit: Optional[int] = None
  filter_fn: Optional[Callable[[Any], bool]] = None
  # Note: Examples are filtered *before* map_fn is applied.
  map_fn: Optional[Callable[[Any], Any]] = None
  cache: bool = False
  shuffle: bool = True
  shuffle_buffer_size: int = 10_000
  num_epochs: Optional[int] = None

  def __call__(self, seed: Optional[PRNGKeyLike] = None) -> tf.data.Dataset:
    dataset_builder = tfds.builder(self.name, data_dir=self.data_dir)

    if self.shuffle and seed is None:
      raise ValueError("Shuffling requires a random seed.")

    rngs = 2 * [[None, None]]
    if seed is not None:
      rng = seed if isinstance(seed, PRNGKey) else jax.random.PRNGKey(seed)
      rng = jax.random.fold_in(
          rng, jax.process_index()
      )  # Derive RNG for this host.
      rngs = list(jax.random.split(rng, 2))

    shuffle_files = self.shuffle
    if self.data_limit is not None:
      # When limiting the number of data points, always take from the same file.
      logging.warning(
          "data_limit is being used in TfdsDataset. This disables file level "
          "shuffling but does not effect example level shuffling."
      )
      shuffle_files = False

    # Following should be jax.random.bits(rngs.pop()) but this will change the
    # resulting seed.
    read_config = tfds.ReadConfig(
        shuffle_seed=int(jax.random.key_data(rngs.pop())[0])
    )
    split = tfds.split_for_jax_process(self.split)
    ds = dataset_builder.as_dataset(
        split=split,
        shuffle_files=shuffle_files,
        read_config=read_config,
        decoders=self.decoders,
    )

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
