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

# TODO(epot): This should go inside orbax/contrib/ or similar.

from __future__ import annotations

import dataclasses

from etils import epath
import jax
from jax.experimental import multihost_utils
from orbax import checkpoint as ocp
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class TFDataHandler(ocp.CheckpointHandler):
  """Handler for tf.data datasets.

  Note: This expect the save and restore envirement to have the same number of
  hosts.
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
