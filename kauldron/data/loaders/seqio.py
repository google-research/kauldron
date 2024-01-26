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


import dataclasses
from typing import Any, Callable, Optional

from grain._src.core import constants
import jax
from kauldron.data.loaders import base
from kauldron.typing import PRNGKey, PRNGKeyLike  # pylint: disable=g-multiple-import
import seqio
import tensorflow as tf


def _get_seqio_dataset(
    split: str,
    task_name: str,
    shuffle: bool = True,
    shuffle_buffer_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
  """Loads SeqIO dataset."""
  # _register_taskcan only be called once.
  if task_name not in seqio.TaskRegistry.names():
    raise ValueError(f"Task {task_name} not found in seqio.TaskRegistry.")

  task = seqio.TaskRegistry.get(task_name)
  return task.get_dataset(
      split=split,
      shuffle=shuffle,
      shuffle_buffer_size=shuffle_buffer_size,
      seed=seed,
      num_epochs=None,
  )


@dataclasses.dataclass(repr=False)
class SeqIO(base.DataLoader):
  """Basic SeqIO dataset loader."""

  # seqio parameters.
  split: str
  task_name: str
  # Post-processing parameters.
  filter_fn: Optional[Callable[[Any], bool]] = None
  # Note: Examples are filtered *before* map_fn is applied.
  map_fn: Optional[Callable[[Any], Any]] = None
  cache: bool = False
  shuffle: bool = True
  shuffle_buffer_size: int = 10_000
  num_epochs: Optional[int] = None
  register: Optional[Any] = None

  def __call__(self, seed: Optional[PRNGKeyLike] = None) -> tf.data.Dataset:
    if self.shuffle and seed is None:
      raise ValueError("Shuffling requires a random seed.")

    if seed is not None:
      rng = seed if isinstance(seed, PRNGKey) else jax.random.PRNGKey(seed)
      rng = jax.random.fold_in(
          rng, jax.process_index()
      )  # Derive RNG for this host.
      dataset_seed = int(jax.random.bits(rng))
    else:
      dataset_seed = None

    # Load tf.Dataset object
    ds = _get_seqio_dataset(
        split=self.split,
        task_name=self.task_name,
        seed=dataset_seed,
        shuffle=self.shuffle,
        shuffle_buffer_size=self.shuffle_buffer_size,
    )

    if self.filter_fn is not None:
      ds = ds.filter(self.filter_fn)

    if self.map_fn is not None:
      ds = ds.map(self.map_fn)

    if self.cache:
      ds = ds.cache()

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

