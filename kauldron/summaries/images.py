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

"""Image summaries."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

import einops
from flax import struct
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Array, Float, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowImages(metrics.Metric):
  """Show image summaries with optional reshaping.

  Attributes:
    images: Key to the images to display.
    num_images: Number of images to collect and display. Default 5.
    vrange: Optional value range of the input images. Used to clip aand then
      rescale the images to [0, 1].
    rearrange: Optional einops string to reshape the images.
    rearrange_kwargs: Optional keyword arguments for the einops reshape.
  """

  images: kontext.Key

  num_images: int = 5
  in_vrange: Optional[tuple[float, float]] = None

  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] | None = None

  @struct.dataclass
  class State(metrics.AutoState["ShowImages"]):
    """Collects the first num_images images."""

    images: Float["n h w #3"] = metrics.truncate_field(
        num_field="parent.num_images"
    )

    @typechecked
    def compute(self) -> Float["n h w #3"]:
      images = super().compute().images
      # always clip to avoid display problems in TB and Datatables
      return np.clip(images, 0.0, 1.0)

  @typechecked
  def get_state(
      self,
      images: Float["..."],
  ) -> ShowImages.State:
    # maybe rearrange and then check shape
    images = _maybe_rearrange(images, self.rearrange, self.rearrange_kwargs)
    check_type(images, Float["n h w #3"])

    # Truncate just as an optimization to avoid unnecessary computations.
    images = images[: self.num_images]

    if self.in_vrange is not None:
      vmin, vmax = self.in_vrange
      images = images.clip(vmin, vmax)
      images = (images - vmin) / (vmax - vmin)

    return self.State(images=images)


def _maybe_rearrange(
    array: Array["..."] | None,
    rearrange: Optional[str] = None,
    rearrange_kwargs: Mapping[str, Any] | None = None,
) -> Array["..."] | None:
  if array is None or rearrange is None:
    return array
  rearrange_kwargs = rearrange_kwargs if rearrange_kwargs is not None else {}

  return einops.rearrange(array, rearrange, **rearrange_kwargs)
