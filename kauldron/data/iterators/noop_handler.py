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

import dataclasses
from typing import Any

from etils import epath
from orbax import checkpoint as ocp
import tensorflow as tf


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
