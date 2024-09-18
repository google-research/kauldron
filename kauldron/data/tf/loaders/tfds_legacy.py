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
from typing import Any, Mapping, Optional

from etils import epath
import jax
from kauldron import kd
from kauldron.data.tf.loaders import with_shuffle_buffer
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass(frozen=True)
class TfdsLegacy(with_shuffle_buffer.WithShuffleBuffer):
  """Simple wrapper around `tfds.load`.

  `kd.data.tf.Tfds` should be preferred as it supports random access nativelly.
  This class is provided for old datasets still using TFRecord format.
  """

  name: str
  _: dataclasses.KW_ONLY
  split: str
  data_dir: epath.PathLike | None = None
  decoders: Optional[Mapping[str, Any]] = None
  read_config: Optional[tfds.ReadConfig] = None

  def ds_for_current_process(self, rng: kd.random.PRNGKey) -> tf.data.Dataset:
    builder = tfds.builder(self.name, data_dir=self.data_dir)

    read_config = self.read_config or tfds.ReadConfig()

    if self.cache:
      # Avoid caching the dataset twice when `cache=True` (this is done by
      # `WithShuffleBuffer`).
      read_config.try_autocache = False

    if self.shuffle:
      # Each process has its own seed.
      seed = rng.fold_in(jax.process_index()).as_seed()
      read_config.shuffle_seed = seed
      # read_config.shuffle_reshuffle_each_iteration = True
      read_config.enable_ordering_guard = False

    split = tfds.split_for_jax_process(self.split)
    return builder.as_dataset(
        split=split,
        shuffle_files=self.shuffle,
        read_config=read_config,
        decoders=self.decoders,
    )
