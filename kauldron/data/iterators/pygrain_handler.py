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

"""PyGrain iterator."""

import dataclasses
from typing import Self

from grain import python as grain
from kauldron.data.iterators import iterators
from orbax import checkpoint as ocp


@dataclasses.dataclass(frozen=True, kw_only=True)
class PyGrainIterator(iterators.Iterator):
  """PyGrain iterator."""
  iter: grain.PyGrainDatasetIterator

  # ================ Implement the checkpoint protocol ================

  def __kd_ocp_handlers__(self) -> ocp.CheckpointHandler:
    return grain.PyGrainCheckpointHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return grain.PyGrainCheckpointSave(self.iter)

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    return grain.PyGrainCheckpointRestore(self.iter)

  def __kd_ocp_restore_post__(
      self, value: grain.PyGrainDatasetIterator
  ) -> Self:
    assert isinstance(value, grain.PyGrainDatasetIterator)
    # Note that like TF, the `self.iter` is mutated in-place, so could return
    # `self` here.
    return PyGrainIterator(source=self.source, iter=value)
