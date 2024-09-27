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

"""Point cloud summaries."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

import einops
from etils import etree
import flax
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Float, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PointCloud:
  """Output type for point cloud summaries."""

  # TODO(klausg): rename to `vertices` and `colors`?
  point_clouds: Float["n 3"]
  point_colors: Optional[Float["n 3"]] = None

  # TODO(epot): replace configs with more structured option
  configs: Optional[Mapping[str, Any]] = None


_DEFAULT_CONFIGS = {
    "camera": {
        "cls": "PerspectiveCamera",
        "near": 1e-4,
    },
    "material": {
        "cls": "PointsMaterial",
        "size": 0.03,
    },
}


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowPointCloud(metrics.Metric):
  """Show a point cloud with optional reshaping."""

  point_clouds: kontext.Key
  point_colors: Optional[kontext.Key] = None
  configs: Optional[Mapping[str, Any]] = None

  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict[str, Any]
  )
  num_points: Optional[int] = None

  # TODO(klausg): use CollectFirstState after adding support for keep_first=None
  @flax.struct.dataclass
  class State(metrics.CollectingState["ShowPointCloud"]):
    """Collecting state that returns PointCloudsData."""

    point_clouds: Float["n 3"]
    point_colors: Optional[Float["n 3"]]

    def compute(self) -> PointCloud:
      results = super().compute()

      check_type(results.point_clouds, Float["n 3"])
      check_type(results.point_colors, Float["n 3"] | None)

      if results.point_clouds.size == 0:
        raise ValueError(
            f"Point cloud summary for {self.parent!r} is an empty array "
            f"point_clouds={etree.spec_like(results.point_clouds)}."
        )

      configs = self.parent.configs or _DEFAULT_CONFIGS
      # To fix write/serialize error with immutable dict (self.configs)
      configs = {k: dict(v) for k, v in configs.items()}

      return PointCloud(
          point_clouds=results.point_clouds,
          point_colors=results.point_colors,
          configs=configs,
      )

  @typechecked
  def get_state(
      self,
      point_clouds: Float,
      point_colors: Optional[Float],
  ) -> ShowPointCloud.State:
    if self.rearrange:
      point_clouds = einops.rearrange(
          point_clouds, self.rearrange, **self.rearrange_kwargs
      )
      if point_colors is not None:
        point_colors = einops.rearrange(
            point_colors, self.rearrange, **self.rearrange_kwargs
        )
    check_type(point_clouds, Float["n 3"])
    check_type(point_colors, Float["n 3"] | None)

    num_points = self.num_points or point_clouds.shape[0]
    point_clouds = point_clouds[:num_points]
    if point_colors is not None:
      point_colors = point_colors[:num_points]

    return self.State(
        point_clouds=point_clouds,
        point_colors=point_colors,
    )
