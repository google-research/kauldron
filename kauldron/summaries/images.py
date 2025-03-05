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
from etils import epy
from flax import struct
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Array, Bool, Float, Integer, UInt8, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np

with epy.lazy_imports():
  import matplotlib.colors  # pylint: disable=g-import-not-at-top
  import mediapy as media  # pylint: disable=g-import-not-at-top
  import tensorflow as tf  # pylint: disable=g-import-not-at-top
  from kauldron.utils import plot_segmentation as segplot  # pylint: disable=g-import-not-at-top


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

    images: Float["n h w #3"] | UInt8["n h w #3"] = metrics.truncate_field(
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
      images: Float["..."] | UInt8["..."],
  ) -> ShowImages.State:
    # maybe rearrange and then check shape
    images = _maybe_rearrange(images, self.rearrange, self.rearrange_kwargs)
    if isinstance(images, UInt8["..."]):
      images = images.astype(np.float32) / 255.0
    check_type(images, Float["n h w #3"])

    # Truncate just as an optimization to avoid unnecessary computations.
    images = images[: self.num_images]

    if self.in_vrange is not None:
      vmin, vmax = self.in_vrange
      images = images.clip(vmin, vmax)
      images = (images - vmin) / (vmax - vmin)

    return self.State(images=images)


# TODO(klausg): The use of rearrange is weird here. maybe move to contrib?
@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowBoxes(metrics.Metric):
  """Show a set of boxes with optional image reshaping.

  Attributes:
    images: Key to the images to display.
    boxes: Key to the boxes to display as `Float["*b k 4"]`. The coordinates of
      each bounding box in `boxes` is encoded as `[y_min, x_min, y_max, x_max]`.
      The bounding box coordinates are floats in `[0.0, 1.0]` relative to the
      width and the height of the underlying image.
    boxes_mask: Optional key to the boxes mask in the form `Bool["*b k 1"]`. The
      mask is used to mask out boxes that should not be displayed.
    num_images: Number of images to collect and display. Default 5.
    num_colors: Number of different colors to use for the boxes. Default 16.
    in_vrange: Optional value range of the input images. Used to clip and then
      rescale the images to [0, 1].
    rearrange: Optional einops string to reshape the images AFTER the boxes have
      been drawn.
    rearrange_kwargs: Optional keyword arguments for the einops reshape.
  """

  images: kontext.Key = kontext.REQUIRED
  boxes: kontext.Key = kontext.REQUIRED
  boxes_mask: Optional[kontext.Key] = None

  num_images: int = 5
  num_colors: int = 16
  in_vrange: Optional[tuple[float, float]] = None

  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] | None = None

  @struct.dataclass
  class State(metrics.AutoState["ShowBoxes"]):
    """Collects the first num_images images and boxes."""

    images: Float["n h w #3"] = metrics.truncate_field(
        num_field="parent.num_images"
    )
    boxes: Float["n k 4"] = metrics.truncate_field(
        num_field="parent.num_images"
    )

    @typechecked
    def compute(self) -> Float["n h w #3"]:
      data = super().compute()
      images, boxes = data.images, data.boxes

      # flatten batch dimensions
      images_shape = images.shape
      images = einops.rearrange(images, "... h w c -> (...) h w c")
      boxes = einops.rearrange(boxes, "... k d -> (...) k d")

      check_type(images, Float["n h w #3"])
      check_type(boxes, Float["n k 4"])

      # draw boxes on images
      colors = _get_uniform_colors(self.parent.num_colors)
      images = tf.image.draw_bounding_boxes(images, boxes, colors)

      # Note: rearrange is applied AFTER the boxes are drawn.
      images = np.reshape(images, images_shape)
      images = _maybe_rearrange(
          images, self.parent.rearrange, self.parent.rearrange_kwargs
      )

      # always clip to avoid display problems in TB and Datatables
      return np.clip(images, 0.0, 1.0)

  @typechecked
  def get_state(
      self,
      images: Float["..."],
      boxes: Float["*b k 4"],
      boxes_mask: Bool["*b k 1"] | None = None,
  ) -> ShowImages.State:
    if boxes_mask is not None:
      boxes = boxes * np.array(boxes_mask, dtype=np.float32)

    # maybe rescale
    if self.in_vrange is not None:
      vmin, vmax = self.in_vrange
      images = images.clip(vmin, vmax)
      images = (images - vmin) / (vmax - vmin)

    return self.State(images=images, boxes=boxes)


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowSegmentations(metrics.Metric):
  """Show a set of segmentations with optional reshaping.

  Attributes:
    segmentations: Key to the segmentations to display.
    num_images: Number of segmentations to collect and display. Default 5.
    entropy: Whether to scale the lightness of the segments in proportion to the
      (normalized) per-pixel entropy of the soft-segmentation.
    rearrange: Optional einops string to reshape the images.
    rearrange_kwargs: Optional keyword arguments for the einops reshape.
  """

  segmentations: kontext.Key

  num_images: int = 5
  entropy: bool = False

  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] | None = None

  @struct.dataclass
  class State(metrics.AutoState["ShowSegmentations"]):
    """Collects the first num_images segmentations."""

    segmentations: Integer["*b h w 1"] | Float["*b h w k"] = (
        metrics.truncate_field(num_field="parent.num_images")
    )

    @typechecked
    def compute(self) -> Float["n h w #3"]:
      segmentations = super().compute().segmentations
      segmentation_images = segplot.plot_segmentation(
          segmentations, entropy=self.parent.entropy
      )
      # always clip to avoid display problems in TB and Datatables
      return np.clip(segmentation_images, 0.0, 1.0)

  @typechecked
  def get_state(
      self,
      segmentations: Float["..."],
  ) -> ShowImages.State:
    # maybe rearrange and then check shape
    segmentations = _maybe_rearrange(
        segmentations, self.rearrange, self.rearrange_kwargs
    )
    check_type(segmentations, Integer["n h w 1"] | Float["n h w k"])
    return self.State(segmentations=segmentations)


# TODO(klausg): Only 8 uses ATM. Move to contrib?
@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowDifferenceImages(metrics.Metric):
  """Show a set of difference images with optional reshaping.

  Attributes:
    images1: Key to the first set of images (from which images2 is subtracted).
    images2: Key to the second set of images (subtracted from images1).
    num_images: Number of images to collect and display. Default 5.
    vrange: Optional value range of the input images. Used to clip and then
      rescale the images to [0, 1].
    cmap: A `pyplot` color map name, to map from 1D value to 3D color.
    rearrange: Optional einops string to reshape the images.
    rearrange_kwargs: Optional keyword arguments for the einops reshape.
  """

  images1: kontext.Key
  images2: kontext.Key

  num_images: int
  vrange: tuple[float, float] | None = None
  cmap: str = "gray"

  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] | None = None

  @struct.dataclass
  class State(metrics.AutoState["ShowDifferenceImages"]):
    """Collects the first num_images images."""

    diff_images: Float["n h w 1"] = metrics.truncate_field(
        num_field="parent.num_images"
    )

    @typechecked
    def compute(self) -> Float["n h w #3"]:
      diff_images = super().compute().diff_images

      # Use the vrange bounds for the colormapping if available.
      if self.parent.vrange is not None:
        vmin, vmax = self.parent.vrange
        diff_vmax = vmax - vmin
        diff_vmin = vmin - vmax
      else:
        diff_vmin, diff_vmax = (None, None)

      # Apply the colormap.
      images = media.to_rgb(
          diff_images[..., 0],
          cmap=self.parent.cmap,
          vmin=diff_vmin,
          vmax=diff_vmax,
      )

      # always clip to avoid display problems in TB and Datatables
      return np.clip(images, 0.0, 1.0)

  @typechecked
  def get_state(
      self,
      images1: Float["..."],
      images2: Float["..."],
  ) -> ShowImages.State:
    # maybe rearrange and then check shape
    images1 = _maybe_rearrange(images1, self.rearrange, self.rearrange_kwargs)
    images2 = _maybe_rearrange(images2, self.rearrange, self.rearrange_kwargs)
    check_type(images1, Float["n h w c"])
    check_type(images1, Float["n h w c"])

    # Truncate just as an optimization to avoid unnecessary computations.
    images1 = images1[: self.num_images]
    images2 = images2[: self.num_images]

    if self.vrange is not None:
      vmin, vmax = self.vrange
      images1 = np.clip(images1, vmin, vmax)
      images2 = np.clip(images2, vmin, vmax)
      images1 = (images1 - vmin) / (vmax - vmin)
      images2 = (images2 - vmin) / (vmax - vmin)

    diff_images = np.abs(images1 - images2)
    diff_images = np.mean(diff_images, axis=-1, keepdims=True)

    return self.State(diff_images=diff_images)


def _maybe_rearrange(
    array: Array["..."] | None,
    rearrange: Optional[str] = None,
    rearrange_kwargs: Mapping[str, Any] | None = None,
) -> Array["..."] | None:
  if array is None:
    return array
  if rearrange is None:  # Auto-flatten images.
    rearrange = "... h w c -> (...) h w c"
  rearrange_kwargs = rearrange_kwargs if rearrange_kwargs is not None else {}

  return einops.rearrange(array, rearrange, **rearrange_kwargs)


def _get_uniform_colors(n_colors: int) -> Array:
  """Get n_colors with uniformly spaced hues."""
  hues = np.linspace(0, 1, n_colors, endpoint=False)
  hsv_colors = np.concatenate(
      (np.expand_dims(hues, axis=1), np.ones((n_colors, 2))), axis=1
  )
  rgb_colors = matplotlib.colors.hsv_to_rgb(hsv_colors)
  return rgb_colors  # rgb_colors.shape = (n_colors, 3)
