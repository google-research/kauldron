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

"""Interface for custom summaries."""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Any, Mapping, Optional

import einops
import flax
import jax
import jax3d.utils.plot_segmentation as segplot
from kauldron import core
from kauldron.typing import Float, Integer, Key, Shape, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import mediapy as media
import numpy as np


Images = Float["*b h w c"] | UInt8["*b h w c"]
Segmentations = Integer["*b h w 1"] | Float["*b h w k"]


class Summary(abc.ABC):
  """Base class for defining non-scalar tensorboard summaries (e.g. images)."""

  def gather_kwargs(self, context: Any) -> dict[str, Any]:
    """Returns the required information from context as a kwargs dict."""
    return core.resolve_kwargs(self, context)


class ImageSummary(Summary, abc.ABC):
  """Base class for image summaries."""

  @abc.abstractmethod
  def get_images(self, **kwargs) -> Images:
    ...

  @typechecked
  def __call__(self, *, context: Any = None, **kwargs) -> Images:
    if context is not None:
      if kwargs:
        raise TypeError(
            "Can either pass context or keyword arguments,"
            f"but got context and {kwargs.keys()}."
        )
      kwargs = core.resolve_kwargs(self, context, func=self.get_images)
    return self.get_images(**kwargs)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ShowImages(ImageSummary):
  """Show a set of images with optional reshaping and resizing."""

  images: Key

  num_images: int
  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict[str, Any]
  )
  width: Optional[int] = None
  height: Optional[int] = None

  def gather_kwargs(self, context: Any) -> dict[str, Images]:
    # optimize gather_kwargs to only return num_images many images
    kwargs = core.resolve_kwargs(self, context)
    images = kwargs["images"]
    if self.rearrange:
      images = einops.rearrange(images, self.rearrange, **self.rearrange_kwargs)
    if not isinstance(images, Float["n h w #3"]):
      raise ValueError(f"Bad shape or dtype: {images.shape} {images.dtype}")

    num_images_per_device = math.ceil(
        self.num_images / jax.local_device_count()
    )
    images = images[:num_images_per_device]

    return {"images": images}

  @typechecked
  def get_images(self, images: Images) -> Float["n _h _w c"]:
    # flatten batch dimensions
    images = einops.rearrange(images, "... h w c -> (...) h w c")
    images = images[: self.num_images]
    # convert to float
    images = media.to_type(images, np.float32)
    # maybe resize
    if (self.width, self.height) != (None, None):
      shape = _get_height_width(self.width, self.height, Shape("h w"))
      images = media.resize_video(images, shape)
    return images


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ShowSegmentations(ImageSummary):
  """Show a set of segmentations with optional reshaping and resizing."""

  segmentations: Key

  num_images: int
  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict[str, Any]
  )
  width: Optional[int] = None
  height: Optional[int] = None
  entropy: Optional[bool] = False

  def gather_kwargs(self, context: Any) -> dict[str, Segmentations]:
    # optimize gather_kwargs to only return num_images many images
    kwargs = core.resolve_kwargs(self, context)
    segmentations = kwargs["segmentations"]
    if self.rearrange:
      segmentations = einops.rearrange(
          segmentations, self.rearrange, **self.rearrange_kwargs
      )
    if not isinstance(segmentations, Segmentations):
      raise ValueError(
          f"Bad shape or dtype: {segmentations.shape} {segmentations.dtype}"
      )

    num_images_per_device = math.ceil(
        self.num_images / jax.local_device_count()
    )
    segmentations = segmentations[:num_images_per_device]

    return {"segmentations": segmentations}

  @typechecked
  def get_images(self, segmentations: Segmentations) -> Float["n _h _w c"]:
    # flatten batch dimensions
    segmentations = einops.rearrange(segmentations, "... h w k -> (...) h w k")
    segmentations = segmentations[: self.num_images]
    segmentation_images = segplot.plot_segmentation(
        segmentations, entropy=self.entropy
    )
    # maybe resize
    if (self.width, self.height) != (None, None):
      shape = _get_height_width(self.width, self.height, Shape("h w"))
      segmentation_images = media.resize_video(segmentation_images, shape)
    return segmentation_images


@typechecked
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
