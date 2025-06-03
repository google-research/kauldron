# Copyright 2025 The kauldron Authors.
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

"""Millstone iterator that implements the Kauldron API."""

import dataclasses
from typing import Any, Self

from kauldron.data.iterators import iterators
from orbax import checkpoint as ocp


@dataclasses.dataclass(frozen=True, kw_only=True)
class MillstoneIterator(iterators.Iterator):
  """Millstone iterator that implements the Kauldron API."""

  iter: remote_dataset.CheckpointableIterator

  def __next__(self) -> Any:
    return next(self.iter)

  # ================ Implements the checkpoint protocol ================
  def __kd_ocp_handlers__(
      self,
  ) -> checkpoint_handler.MillstoneCheckpointHandler:
    return checkpoint_handler.MillstoneCheckpointHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return checkpoint_handler.MillstoneCheckpointSave(self.iter)

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    return checkpoint_handler.MillstoneCheckpointRestore(self.iter)

  def __kd_ocp_restore_post__(
      self, value: remote_dataset.CheckpointableIterator
  ) -> Self:
    assert isinstance(value, remote_dataset.CheckpointableIterator)
    return MillstoneIterator(source=self.source, iter=value)
