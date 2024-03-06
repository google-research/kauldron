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

"""Orbax no-op handler."""

# TODO(epot): This should go inside orbax/contrib/ or similar.

from __future__ import annotations

import collections.abc
import dataclasses
from typing import Any

from etils import epath
from kauldron.data.iterators import iterators
from orbax import checkpoint as ocp
import tensorflow as tf


@dataclasses.dataclass(frozen=True, kw_only=True)
class NonCheckpointableIterator(iterators.Iterator):
  """Handler that is not-checkpointable."""

  iter: collections.abc.Iterator[Any]

  def __next__(self):
    return next(self.iter)

  # ================ Implement the checkpoint protocol ================

  def __kd_ocp_handlers__(self) -> ocp.CheckpointHandler:
    return NoopHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return NoopArg(self)

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    # Note: The `ds_iter` is mutated in-place !!! (because TF)
    return NoopArg(self)


class NoopHandler(ocp.CheckpointHandler):
  """Handler that forward the state as-is."""

  def save(self, directory: epath.Path, args: NoopArg):
    pass

  def restore(self, directory: epath.Path, args: NoopArg) -> tf.data.Iterator:
    return args.value


@ocp.args.register_with_handler(NoopHandler, for_save=True, for_restore=True)
@dataclasses.dataclass()
class NoopArg(ocp.args.CheckpointArgs):
  value: Any
