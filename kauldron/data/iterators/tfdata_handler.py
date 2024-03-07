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

"""Orbax tf.data handler."""

from __future__ import annotations

import dataclasses
from typing import Self

from etils import epath
import jax
from jax.experimental import multihost_utils
from kauldron.data.iterators import iterators
from orbax import checkpoint as ocp
import tensorflow as tf
import tensorflow_datasets as tfds


@dataclasses.dataclass(frozen=True, kw_only=True)
class TFDataIterator(iterators.Iterator):
  """Checkpointable `tf.data` iterator."""
  iter: tf.data.Iterator

  def __next__(self):
    ex = next(self.iter)
    ex = tfds.as_numpy(ex)
    return ex

  # ================ Implement the checkpoint protocol ================

  def __kd_ocp_handlers__(self) -> ocp.CheckpointHandler:
    return TFDataHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return TFDataArg(self.iter)

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    # Note: The `ds_iter` is mutated in-place !!! (because TF)
    return TFDataArg(self.iter)

  def __kd_ocp_restore_post__(self, value: tf.data.Iterator) -> Self:
    assert isinstance(value, tf.data.Iterator)
    # In theory, `self.iter is value`, so could return `self` directly
    return TFDataIterator(source=self.source, iter=value)


# TODO(epot): `TFDataHandler` should go inside orbax/contrib/ or similar.


@dataclasses.dataclass(frozen=True, kw_only=True)
class TFDataHandler(ocp.CheckpointHandler):
  """Handler for tf.data datasets.

  * This expect the save and restore envirement to have the same number
    of hosts (one checkpoint is saved per host).
  * `tf.train.Checkpoint` mutate the iterator in-place.
  """
  filename: str = 'tfdata'

  @property
  def dirname(self):
    return f'process_{jax.process_index()}-of-{jax.process_count()}'

  def save(self, directory: epath.Path, args: TFDataArg):
    directory /= self.dirname
    directory.mkdir()

    ckpt = tf.train.Checkpoint(it=args.it)
    ckpt.write(directory / self.filename)

    # Not sure if required, but was in `t5x` codebase
    multihost_utils.sync_global_devices('TfDataHandler:save')

  def restore(self, directory: epath.Path, args: TFDataArg) -> tf.data.Iterator:
    directory /= self.dirname

    ckpt = tf.train.Checkpoint(it=args.it)
    status = ckpt.read(directory / self.filename)
    status.assert_consumed()
    return args.it


@ocp.args.register_with_handler(TFDataHandler, for_save=True, for_restore=True)
@dataclasses.dataclass
class TFDataArg(ocp.args.CheckpointArgs):
  it: tf.data.Iterator
