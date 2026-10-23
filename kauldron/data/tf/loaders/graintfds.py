# Copyright 2026 The kauldron Authors.
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
from kauldron.data import pipelines
from kauldron.data.tf import base
from kauldron.ktyping import PRNGKey
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
  cache: bool = False  # pyrefly: ignore[bad-override]

  # Sampler parameters
  shuffle: bool = True
  num_epochs: Optional[int] = None

  _supports_symbolic_checkpoint: ClassVar[bool] = True  # pyrefly: ignore[bad-override]

  def ds_for_current_process(self, rng: PRNGKey) -> tf.data.Dataset:
    source = grain.TfdsDataSource.from_name(
        self.name,
        split=self.split,
        data_dir=self.data_dir,
        decoders=self.decoders,
        cache=self.cache,
    )

    # Drop the shard remainder only when `batch_drop_remainder` is `DROP` so
    # `KEEP`, `PAD`, and `PAD_WITH_MASK` do not discard examples during
    # multi-host sharding.
    sampler = grain.TfDefaultIndexSampler(
        num_records=len(source),
        shuffle=self.shuffle,
        seed=random.random_seed(rng),
        shard_options=grain.ShardByJaxProcess(
            drop_remainder=(
                self.batch_drop_remainder == pipelines.DropRemainder.DROP
            )
        ),
        num_epochs=self.num_epochs,
        shard_before_shuffle=False,
    )

    data_loader = grain.TfDataLoader(source=source, sampler=sampler)

    return data_loader.as_dataset(start_index=grain.FirstIndex())
