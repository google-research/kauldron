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

"""TfGrain iterator."""

import dataclasses
from typing import Self

from grain import tensorflow as grain
from kauldron.data.iterators import iterators
from orbax import checkpoint as ocp


# `grain.TfGrainCheckpointHandler` do not inherit from `ocp.CheckpointHandler`,
# making some orbax condition fail.
class TfGrainCheckpointHandler(
    grain.OrbaxCheckpointHandler, ocp.CheckpointHandler
):
  pass


# And we need to recreate the handlers so the registration system do not
# complain.
@ocp.args.register_with_handler(TfGrainCheckpointHandler, for_save=True)
class TfGrainCheckpointSave(grain.TfGrainCheckpointSave):
  pass


@ocp.args.register_with_handler(TfGrainCheckpointHandler, for_restore=True)  # pytype:disable=wrong-arg-types
class TfGrainCheckpointRestore(grain.TfGrainCheckpointRestore):
  pass


@ocp.args.register_with_handler(
    TfGrainCheckpointHandler, for_save=True, for_restore=True
)
@dataclasses.dataclass
class TfGrainArg(ocp.args.CheckpointArgs):
  item: grain.TfGrainDatasetIterator


@dataclasses.dataclass(frozen=True, kw_only=True)
class TfGrainIterator(iterators.Iterator):
  """TfGrain iterator."""

  iter: grain.TfGrainDatasetIterator

  def __next__(self):
    return next(self.iter)

  # ================ Implement the checkpoint protocol ================

  def __kd_ocp_handlers__(self) -> TfGrainCheckpointHandler:
    return TfGrainCheckpointHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return TfGrainCheckpointSave(self.iter)

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    return TfGrainCheckpointRestore(self.iter)

  def __kd_ocp_restore_post__(
      self, value: grain.TfGrainDatasetIterator
  ) -> Self:
    assert isinstance(value, grain.TfGrainDatasetIterator)
    return TfGrainIterator(source=self.source, iter=value)
