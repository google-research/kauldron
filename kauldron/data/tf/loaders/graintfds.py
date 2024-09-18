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
from typing import Any, ClassVar, Mapping, Optional

from etils import epath
import grain.tensorflow as grain
from kauldron import random
from kauldron.data.tf import base
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class Tfds(base.TFDataPipeline):
  """Basic TFDS dataset loader (without transformations).

  This uses PyGrain and global shuffling, so only support ArrayRecord datasets.
  """

  name: str
  _: dataclasses.KW_ONLY
  split: str
  data_dir: epath.PathLike | None = None
  decoders: Optional[Mapping[str, Any]] = None
  cache: bool = False

  # Sampler parameters
  shuffle: bool = True
  shard_drop_remainder: bool = True
  num_epochs: Optional[int] = None

  _supports_symbolic_checkpoint: ClassVar[bool] = True

  def ds_for_current_process(self, rng: random.PRNGKey) -> tf.data.Dataset:
    source = grain.TfdsDataSource.from_name(
        self.name,
        split=self.split,
        data_dir=self.data_dir,
        decoders=self.decoders,
        cache=self.cache,
    )

    sampler = grain.TfDefaultIndexSampler(
        num_records=len(source),
        shuffle=self.shuffle,
        seed=rng.as_seed(),
        shard_options=grain.ShardByJaxProcess(
            drop_remainder=self.shard_drop_remainder
        ),
        num_epochs=self.num_epochs,
        shard_before_shuffle=False,
    )

    data_loader = grain.TfDataLoader(source=source, sampler=sampler)

    return data_loader.as_dataset(start_index=grain.FirstIndex())
