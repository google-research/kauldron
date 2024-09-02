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


# `grain.PyGrainCheckpointHandler` do not inherit from `ocp.CheckpointHandler`,
# making some orbax condition fail.
class PyGrainCheckpointHandler(
    grain.PyGrainCheckpointHandler, ocp.CheckpointHandler
):
  pass


# And we need to recreate the handlers so the registration system do not
# complain.
@ocp.args.register_with_handler(PyGrainCheckpointHandler, for_save=True)
class PyGrainCheckpointSave(grain.PyGrainCheckpointSave):
  pass


@ocp.args.register_with_handler(PyGrainCheckpointHandler, for_restore=True)  # pytype:disable=wrong-arg-types
class PyGrainCheckpointRestore(grain.PyGrainCheckpointRestore):
  pass


@dataclasses.dataclass(frozen=True, kw_only=True)
class PyGrainIterator(iterators.Iterator):
  """PyGrain iterator."""
  iter: grain.DatasetIterator

  def __next__(self):
    return next(self.iter)

  # ================ Implement the checkpoint protocol ================

  def __kd_ocp_handlers__(self) -> PyGrainCheckpointHandler:
    return PyGrainCheckpointHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return PyGrainCheckpointSave(self.iter)

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    return PyGrainCheckpointRestore(self.iter)

  def __kd_ocp_restore_post__(self, value: grain.DatasetIterator) -> Self:
    assert isinstance(value, grain.DatasetIterator)
    # Note that like TF, the `self.iter` is mutated in-place, so could return
    # `self` here.
    return PyGrainIterator(source=self.source, iter=value)
