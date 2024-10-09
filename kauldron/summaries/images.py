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

"""Histogram summaries."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, TypeVar

import einops
from flax import struct
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Bool, Float, Shape, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import mediapy as media
import numpy as np

_MetricT = TypeVar("_MetricT")


@struct.dataclass
class CollectImages(metrics.AutoState[_MetricT]):
  """Collects the first num_images images and optionally resizes them."""

  images: Float["n h w #3"] = metrics.truncate_field(num_field="num_images")

  num_images: int = metrics.static_field()
  width: int | None = metrics.static_field(default=None)
  height: int | None = metrics.static_field(default=None)

  @typechecked
  def compute(self):
    images = super().compute().images
    check_type(images, Float["n h w #3"])
    if self.width is not None and self.height is not None:
      shape = _get_height_width(self.width, self.height, Shape("h w"))
      images = media.resize_video(images, shape)

    # always clip to avoid display problems in TB and Datatables
    return np.clip(images, 0.0, 1.0)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ShowImages(metrics.Metric):
  """Show image summaries with optional reshaping, masking, and resizing."""

  images: kontext.Key
  masks: Optional[kontext.Key] = None

  num_images: int
  rearrange: Optional[str] = None
  rearrange_kwargs: Optional[Mapping[str, Any]] = None
  width: Optional[int] = None
  height: Optional[int] = None
  in_vrange: Optional[tuple[float, float]] = None
  mask_color: float | tuple[float, float, float] = 0.5

  @struct.dataclass
  class State(CollectImages["ShowImages"]):
    pass

  @typechecked
  def get_state(
      self,
      images: Float["*b h w #3"],
      masks: Optional[Bool["*b h w 1"]] = None,
  ) -> ShowImages.State:
    images = self._rearrange(images)
    masks = self._rearrange(masks)

    if self.in_vrange is not None:
      vmin, vmax = self.in_vrange
      images = (images - vmin) / (vmax - vmin)

    if masks is not None:
      images = images.at[masks[..., 0]].set(self.mask_color)

    return self.State(
        num_images=self.num_images,
        width=self.width,
        height=self.height,
        images=images,
    )

  def _rearrange(self, img_or_mask):
    if self.rearrange and img_or_mask is not None:
      rearrange_kwargs = self.rearrange_kwargs or {}
      img_or_mask = einops.rearrange(
          img_or_mask, self.rearrange, **rearrange_kwargs
      )
    # flatten batch dimensions
    if img_or_mask is not None:
      img_or_mask = einops.rearrange(img_or_mask, "... h w c -> (...) h w c")
      # truncate to num_images. Just an optimization to avoid unnecessary
      # computation.
      img_or_mask = img_or_mask[: self.num_images]
    return img_or_mask


def _get_height_width(
    width: Optional[int], height: Optional[int], shape: tuple[int, int]
) -> tuple[int, int]:
  """Returns (width, height) given optional parameters and image shape."""
  h, w = shape
  if width and height:
    return height, width
  if width and not height:
    return int(width * (h / w) + 0.5), width
  if height and not width:
    return height, int(height * (w / h) + 0.5)
  return shape
