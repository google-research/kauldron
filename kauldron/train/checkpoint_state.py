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

"""Checkpoint state."""

from __future__ import annotations

import typing

from kauldron import checkpoints
from kauldron.data import iterators
from kauldron.train import train_step
from kauldron.utils import chrono_utils


class _CheckpointState(typing.NamedTuple):
  """Checkpoint state (intermediate class).

  Usually it would be good practice to make this a `dataclass(kw_only=True)`.
  However in this case, it's convenient to be able to use it like a
  `NamedTuple`, like:

  ```python
  state, timer, ds_iter = ckptr.restore(CheckpointState(state, timer, ds_iter))
  ```

  The indirection (`class CheckpointState(_CheckpointState)`) is required
  because Python doesn't support multi-inheritance with `NamedTuple`.
  See: https://github.com/python/cpython/issues/116241
  """

  train_state: train_step.TrainState
  timer: chrono_utils.Chrono
  ds_iter: iterators.Iterator

  # `train_state` is saved as the default name
  DEFAULT_ITEM = 'train_state'


class CheckpointState(
    _CheckpointState, checkpoints.items.TopLevelCheckpointItem
):
  """Checkpoint state."""

  pass
