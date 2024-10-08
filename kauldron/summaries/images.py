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

_MetricT = TypeVar("_MetricT", bound="ImageSummaryBase")


@dataclasses.dataclass(kw_only=True, frozen=True)
class ImageSummaryBase(metrics.Metric):
  """Base class for image summaries.

  Subclasses must implement the get_state method which should make use of
  self._rearrange and return a State object. See ShowImages for an example.

  Attributes:
    rearrange: Optional einops string specifying how to reshape the data before
      aggregating and displaying.
    rearrange_kwargs: Optional keyword arguments to pass to einops.rearrange.
    width: Optional width to resize the images to.
    height: Optional height to resize the images to.
    num_images: Number of images to collect and display.
    cmap: Optional colormap to use for displaying single-channelimages.
    mask_color: Color (or grayscale value) to use for indicating masked pixels.
  """

  rearrange: Optional[str] = None
  rearrange_kwargs: Optional[Mapping[str, Any]] = None

  width: Optional[int] = None
  height: Optional[int] = None

  num_images: int

  cmap: str | None = None
  mask_color: float | tuple[float, float, float] = 0.5

  @struct.dataclass
  class State(metrics.AutoState[_MetricT]):
    """Collects the first num_images images and optionally resizes them."""

    images: Float["n h w #3"] = metrics.truncate_field(
        num_field="parent.num_images"
    )
    masks: Bool["n h w 1"] | None = metrics.truncate_field(
        num_field="parent.num_images"
    )

    @typechecked
    def compute(self) -> Float["n h w #3"]:
      data = super().compute()
      images, masks = data.images, data.masks
      check_type(images, Float["n h w #3"])
      check_type(masks, Bool["n h w 1"] | None)

      width, height = self.parent.width, self.parent.height
      if width is not None and height is not None:
        shape = _get_height_width(width, height, Shape("h w"))
        images = media.resize_video(images, shape)

      if self.parent.cmap is not None:
        check_type(images, Float["n h w 1"])
        images = media.to_rgb(images, cmap=self.parent.cmap)

      if masks is not None:
        images = images.at[masks[..., 0]].set(self.parent.mask_color)

      # always clip to avoid display problems in TB and Datatables
      return np.clip(images, 0.0, 1.0)

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


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowImages(ImageSummaryBase):
  """Show image summaries with optional reshaping, masking, and resizing.

  Attributes:
    images: Key to the images to display.
    masks: Optional key to a boolean array indicating masked pixels.
    in_vrange: Optional value range of the input images. Used to clip aand then
      rescale the images to [0, 1].
  """

  images: kontext.Key
  masks: Optional[kontext.Key] = None

  in_vrange: Optional[tuple[float, float]] = None

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
      images = images.clip(vmin, vmax)
      images = (images - vmin) / (vmax - vmin)

    return self.State(
        images=images,
        masks=masks,
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ShowDifferenceImages(ImageSummaryBase):
  """Show a set of difference images with optional reshaping and resizing.

  Computes abs(images1 - images2).mean(axis=-1) normalized to [0, 1].

  Attributes:
    images1: Key to the first set of images.
    images2: Key to the second set of images.
    masks: Optional key to a boolean array indicating masked pixels.
    vrange: Value range of the input images. Used to clip input images and
      rescale the difference images to [0, 1].
  """

  images1: kontext.Key
  images2: kontext.Key
  masks: Optional[kontext.Key] = None

  vrange: tuple[float, float]

  @typechecked
  def get_state(
      self,
      images1: Float["*b h w c"],
      images2: Float["*b h w c"],
      masks: Optional[Bool["*b h w 1"]] = None,
  ) -> ShowDifferenceImages.State:
    images1 = self._rearrange(images1)
    images2 = self._rearrange(images2)
    masks = self._rearrange(masks)

    # Compute absolute difference and mean across channels
    vmin, vmax = self.vrange
    images = np.abs(np.clip(images1, vmin, vmax) - np.clip(images2, vmin, vmax))
    images = np.mean(images, axis=-1, keepdims=True) / (vmax - vmin)

    return self.State(
        images=images,
        masks=masks,
    )


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
