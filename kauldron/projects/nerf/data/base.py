# Copyright 2023 The kauldron Authors.
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

from grain import python as grain
from kauldron import kd
from kauldron.projects.nerf.core import structs


@dataclasses.dataclass(frozen=True, kw_only=True)
class Pipeline(kd.data.PyGrainPipeline):
  """Small wrapper around `PyGrainPipeline`.

  Only to allow a more flat structure in the config:

  * `Pipeline(scene=Blender(), data_source=RaySampler())`

  vs

  * `Pipeline(data_source=RaySampler(scene=Blender()))`
  """

  scene: SceneBuilder

  def __post_init__(self):
    # Future proof in case base class implement `__post_init__` one day
    if hasattr(super(), '__post_init__'):
      super().__post_init__()  # pylint: disable=attribute-error

    new_data_source = dataclasses.replace(
        self.data_source,
        scene_builder=self.scene,
        seed=self.seed,
    )
    object.__setattr__(self, 'data_source', new_data_source)


@dataclasses.dataclass(frozen=True, kw_only=True)
class DataSource(grain.RandomAccessDataSource[structs.Batch]):
  """Nerf data source."""

  scene_builder: SceneBuilder = None
  seed: int | None = None

  def __len__(self) -> int:
    """Returns the total number of records in the data source."""
    raise NotImplementedError('Abstract method')

  def __getitem__(self, record_key: int) -> structs.Batch:  # pytype: disable=signature-mismatch
    raise NotImplementedError('Abstract method')

  @functools.cached_property
  def scene(self) -> structs.Batch:
    return self.scene_builder.scene

  @functools.cached_property
  def batch(self) -> structs.Batch:  # -> structs.Batch['*n h w']
    return self.scene_builder.scene.batch

  @functools.cached_property
  def flat_batch(self) -> structs.Batch:  # -> structs.Batch['(*n h w)']
    return self.scene_builder.scene.flat_batch


# TODO(epot): Cache results among ecolab reloads !!!
@dataclasses.dataclass(frozen=True, kw_only=True)
class SceneBuilder(abc.ABC):
  """Blender dataset."""

  @functools.cached_property
  @abc.abstractmethod
  def scene(self) -> structs.Scene:
    raise NotImplementedError()
