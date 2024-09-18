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

"""SeqIO dataset loader."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, ClassVar, Optional

from etils import epy
import jax
from kauldron import random
from kauldron.data.tf import base
import tensorflow as tf

with epy.lazy_imports(
    error_callback=(
        "seqio requires adding `//third_party/py/seqio` to your trainer."
    )
):
  import seqio  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


@dataclasses.dataclass(frozen=True, kw_only=True)
class _SeqIO(base.TFDataPipeline):
  """Basic SeqIO dataset loader.

  Attributes:
    split: Dataset split
    shuffle: Whether to shuffle the dataset.
    num_epochs: Number of epochs to load. None for infinite dataset.
    register: Additional function, module,... that allow the `konfig.resolve` to
      register the seqio task or mixture. Only used to bind the register
      dependencies to the config.
  """

  # seqio parameters.
  split: str
  shuffle: bool = True
  num_epochs: Optional[int] = None
  register: Optional[Any] = None

  _supports_symbolic_checkpoint: ClassVar[bool] = False

  @property
  def shard_info(self) -> seqio.ShardInfo:
    return seqio.ShardInfo(
        num_shards=jax.process_count(),
        index=jax.process_index(),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SeqIOTask(_SeqIO):
  """SeqIO task.

  Attributes:
    name: Task name
  """

  name: str
  shuffle_buffer_size: Optional[int] = None

  def ds_for_current_process(self, rng: random.PRNGKey) -> tf.data.Dataset:
    if self.name not in seqio.TaskRegistry.names():
      raise ValueError(f"Task {self.name!r} not found in seqio.TaskRegistry.")

    task = seqio.TaskRegistry.get(self.name)
    return task.get_dataset(
        split=self.split,
        shuffle=self.shuffle,
        shuffle_buffer_size=self.shuffle_buffer_size,
        seed=rng.as_seed(),
        shard_info=self.shard_info,
        num_epochs=self.num_epochs,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SeqIOMixture(_SeqIO):
  """SeqIO mixture.

  Attributes:
    name: Mixture name
    sequence_length: Forwarded to `seqio.Mixture.get_dataset`
  """

  name: str

  sequence_length: Optional[Mapping[str, int]] = None

  def ds_for_current_process(self, rng: random.PRNGKey) -> tf.data.Dataset:
    if self.name not in seqio.MixtureRegistry.names():
      raise ValueError(f"Task {self.name} not found in seqio.MixtureRegistry.")

    mixture = seqio.MixtureRegistry.get(self.name)
    return mixture.get_dataset(
        sequence_length=self.sequence_length,
        split=self.split,
        shuffle=self.shuffle,
        seed=rng.as_seed(),
        num_epochs=self.num_epochs,
        shard_info=self.shard_info,
    )
