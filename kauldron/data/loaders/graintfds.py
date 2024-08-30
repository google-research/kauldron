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

"""Grain TFDS dataset loader."""

import dataclasses
from typing import Any, Mapping, Optional

from etils import epath
import grain.tensorflow as grain
import jax
from kauldron import random
from kauldron.data.loaders import base
from kauldron.typing import PRNGKeyLike  # pylint: disable=g-multiple-import,g-importing-member
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass
class GrainTfds(base.DataLoader):
  """Basic Grain TFDS dataset loader (without transformations)."""

  # grain.TfdsDataSource parameters
  name: str
  split: str = "train"
  data_dir: epath.PathLike = (
      tfds.core.constants.DATA_DIR
  )
  decoders: Optional[Mapping[str, Any]] = None
  cache: bool = False

  # Sampler parameters
  shuffle: bool = True
  shard_options: grain.ShardOptions = grain.NoSharding()
  num_epochs: Optional[int] = None

  # Loader Parameters

  def __call__(self, seed: Optional[PRNGKeyLike] = None) -> tf.data.Dataset:
    split = tfds.split_for_jax_process(self.split)
    source = grain.TfdsDataSource.from_name(
        self.name,
        split=split,
        data_dir=self.data_dir,
        decoders=self.decoders,
        cache=self.cache,
    )

    if self.shuffle and seed is None:
      raise ValueError("Shuffling requires a random seed.")

    if seed is not None:
      with jax.transfer_guard("allow"):
        rng = random.PRNGKey(seed)
        rng = rng.fold_in(jax.process_index())  # Derive RNG for this host.
        seed = rng.as_seed()
    else:
      seed = None

    sampler = grain.TfDefaultIndexSampler(
        num_records=len(source),
        shuffle=self.shuffle,
        seed=seed,
        shard_options=self.shard_options,
        num_epochs=self.num_epochs,
    )

    data_loader = grain.TfDataLoader(source=source, sampler=sampler)

    return data_loader.as_dataset(start_index=grain.FirstIndex())
