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

"""Base data source."""

from __future__ import annotations

import abc
import dataclasses
import functools

from projects.nerf.core import structs


@dataclasses.dataclass(frozen=True, kw_only=True)
class SceneBuilder(abc.ABC):
  """Base class to build a scene."""
  flatten: bool

  @functools.cached_property
  @abc.abstractmethod
  def scene(self) -> structs.Scene:
    raise NotImplementedError()

  def __call__(self) -> structs.Batch:
    batch = self.scene.batch

    if self.flatten:
      return batch.flatten()
    else:
      return batch
