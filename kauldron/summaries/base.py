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

"""Interface for custom summaries."""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Any, Mapping, Optional

import einops
from etils import epy
import flax
import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron.typing import Array, Bool, Float, Integer, Shape, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import matplotlib
import mediapy as media
import numpy as np
import sklearn.decomposition
import tensorflow as tf

with epy.lazy_imports():
  import jax3d.utils.plot_segmentation as segplot  # pylint: disable=g-import-not-at-top


Images = Float["*b h w c"] | UInt8["*b h w c"]
Masks = Bool["*b h w 1"]
Segmentations = Integer["*b h w 1"] | Float["*b h w k"]
Boxes = Float["*b k #4"]
BoxesMask = Bool["*b k 1"]


class Summary(abc.ABC):
  """Base class for defining non-scalar tensorboard summaries (e.g. images)."""

  def gather_kwargs(self, context: Any) -> dict[str, Any]:
    """Returns the required information from context as a kwargs dict."""
    return kontext.resolve_from_keyed_obj(context, self)


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
      kwargs = kontext.resolve_from_keyed_obj(
          context, self, func=self.get_images
      )
    return self.get_images(**kwargs)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class HistogramSummary(Summary):
  """Basic histogram summary."""

  tensor: kontext.Key
  num_buckets: int = 30

  def get_tensor(self, tensor: Array["..."]) -> tuple[int, Array["n"]]:
    return self.num_buckets, tensor.flatten()


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ShowImages(ImageSummary):
  """Show a set of images with optional reshaping and resizing."""

  images: kontext.Key
  masks: Optional[kontext.Key] = None

  num_images: int
  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict[str, Any]
  )
  width: Optional[int] = None
  height: Optional[int] = None
  in_vrange: Optional[tuple[float, float]] = None
  mask_color: float | tuple[float, float, float] = 0.5

  def gather_kwargs(self, context: Any) -> dict[str, Images | Masks]:
    # optimize gather_kwargs to only return num_images many images
    kwargs = kontext.resolve_from_keyed_obj(context, self)
    images = kwargs["images"]
    masks = kwargs.get("masks", None)
    if self.rearrange:
      images = einops.rearrange(images, self.rearrange, **self.rearrange_kwargs)
    images = images.astype(jnp.float32)
    if not isinstance(images, Float["n h w #3"]):
      raise ValueError(f"Bad shape or dtype: {images.shape} {images.dtype}")

    num_images_per_device = math.ceil(
        self.num_images / jax.local_device_count()
    )
    images = images[:num_images_per_device]
    if masks is not None:
      if not isinstance(masks, Bool["n h w 1"]):
        raise ValueError(
            f"Bad mask shape or dtype: {masks.shape} {masks.dtype}"
        )
      masks = masks[:num_images_per_device]

    return {"images": images, "masks": masks}

  @typechecked
  def get_images(
      self, images: Images, masks: Optional[Masks] = None
  ) -> Float["n _h _w _c"]:
    # flatten batch dimensions
    images = einops.rearrange(images, "... h w c -> (...) h w c")
    images = np.array(images[: self.num_images])
    # maybe rescale
    if self.in_vrange is not None:
      vmin, vmax = self.in_vrange
      images = (images - vmin) / (vmax - vmin)
    # convert to float
    images = media.to_type(images.astype(jnp.float32), np.float32)

    if masks is not None:
      masks = einops.rearrange(masks, "... h w c -> (...) h w c")
      masks = np.array(masks[: self.num_images, :, :, 0])
      images[masks] = self.mask_color

    # always clip to avoid display problems in TB and Datatables
    images = np.clip(images, 0.0, 1.0)
    # maybe resize
    if (self.width, self.height) != (None, None):
      shape = _get_height_width(self.width, self.height, Shape("h w"))
      images = media.resize_video(images, shape)
    return images


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ShowDifferenceImages(ImageSummary):
  """Show a set of difference images with optional reshaping and resizing."""

  images1: kontext.Key
  images2: kontext.Key
  masks: Optional[kontext.Key] = None

  num_images: int
  vrange: tuple[float, float]
  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict[str, Any]
  )
  width: Optional[int] = None
  height: Optional[int] = None
  cmap: str | None = None
  mask_color: float | tuple[float, float, float] = 0.5

  def gather_kwargs(self, context: Any) -> dict[str, Images | Masks]:
    # optimize gather_kwargs to only return num_images many images
    kwargs = kontext.resolve_from_keyed_obj(context, self)
    images1, images2 = kwargs["images1"], kwargs["images2"]
    masks = kwargs.get("masks", None)
    if self.rearrange:
      images1 = einops.rearrange(
          images1, self.rearrange, **self.rearrange_kwargs
      )
      images2 = einops.rearrange(
          images2, self.rearrange, **self.rearrange_kwargs
      )

    images1 = images1.astype(jnp.float32)
    images2 = images2.astype(jnp.float32)
    if not isinstance(images1, Float["n h w #3"]):
      raise ValueError(f"Bad shape or dtype: {images1.shape} {images1.dtype}")
    if not isinstance(images2, Float["n h w #3"]):
      raise ValueError(f"Bad shape or dtype: {images2.shape} {images2.dtype}")

    num_images_per_device = math.ceil(
        self.num_images / jax.local_device_count()
    )
    images1 = images1[:num_images_per_device]
    images2 = images2[:num_images_per_device]
    if masks is not None:
      if not isinstance(masks, Bool["n h w 1"]):
        raise ValueError(
            f"Bad mask shape or dtype: {masks.shape} {masks.dtype}"
        )
      masks = masks[:num_images_per_device]

    return {"images1": images1, "images2": images2, "masks": masks}

  @typechecked
  def get_images(
      self, images1: Images, images2: Images, masks: Optional[Masks] = None
  ) -> Float["n _h _w _c"]:
    # flatten batch dimensions
    images1 = einops.rearrange(images1, "... h w c -> (...) h w c")
    images2 = einops.rearrange(images2, "... h w c -> (...) h w c")
    images1 = np.array(images1[: self.num_images])
    images2 = np.array(images2[: self.num_images])
    # convert to float
    images1 = media.to_type(images1, np.float32)
    images2 = media.to_type(images2, np.float32)

    # Compute absolute difference and mean across channels
    vmin, vmax = self.vrange
    images = np.abs(np.clip(images1, vmin, vmax) - np.clip(images2, vmin, vmax))
    images = np.mean(images, axis=-1, keepdims=True)

    # Normalize difference image to 0-1 and color.
    cmap = self.cmap if self.cmap else "gray"
    images = media.to_rgb(images[..., 0], cmap=cmap, vmin=0, vmax=vmax - vmin)

    if masks is not None:
      masks = einops.rearrange(masks, "... h w c -> (...) h w c")
      masks = np.array(masks[: self.num_images, :, :, 0])
      images[masks] = self.mask_color

    # always clip to avoid display problems in TB and Datatables
    images = np.clip(images, 0.0, 1.0)
    # maybe resize
    if (self.width, self.height) != (None, None):
      shape = _get_height_width(self.width, self.height, Shape("h w"))
      images = media.resize_video(images, shape)
    return images


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ShowSegmentations(ImageSummary):
  """Show a set of segmentations with optional reshaping and resizing."""

  segmentations: kontext.Key

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
    kwargs = kontext.resolve_from_keyed_obj(context, self)
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


def _get_uniform_colors(n_colors: int) -> Array:
  """Get n_colors with uniformly spaced hues."""
  hues = np.linspace(0, 1, n_colors, endpoint=False)
  hsv_colors = np.concatenate(
      (np.expand_dims(hues, axis=1), np.ones((n_colors, 2))), axis=1
  )
  rgb_colors = matplotlib.colors.hsv_to_rgb(hsv_colors)
  return rgb_colors  # rgb_colors.shape = (n_colors, 3)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ShowBoxes(ImageSummary):
  """Show a set of boxes with optional image reshaping and resizing."""

  images: kontext.Key
  boxes: kontext.Key
  boxes_mask: Optional[kontext.Key] = None

  num_images: int
  num_colors: int = 16
  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict[str, Any]
  )
  width: Optional[int] = None
  height: Optional[int] = None
  in_vrange: Optional[tuple[float, float]] = None

  def gather_kwargs(self, context: Any) -> dict[str, Segmentations]:
    # NOTE: moved all logic to get_images(), which is executed on CPU.
    kwargs = kontext.resolve_from_keyed_obj(context, self)
    images = kwargs["images"]
    boxes = kwargs["boxes"]
    boxes_mask = kwargs["boxes_mask"]

    return {
        "images": images[: self.num_images],
        "boxes": boxes[: self.num_images],
        "boxes_mask": (
            boxes_mask[: self.num_images] if boxes_mask is not None else None
        ),
    }

  @typechecked
  def get_images(
      self, images: Images, boxes: Boxes, boxes_mask: BoxesMask | None
  ) -> Float["n _h _w _c"]:
    # convert to numpy
    images = np.array(images, dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    if boxes_mask is not None:
      boxes = boxes * np.array(boxes_mask, dtype=np.float32)

    # first draw boxes on images, before any additional processing takes place
    images_shape = images.shape
    images = einops.rearrange(images, "... h w c -> (...) h w c")
    boxes = einops.rearrange(boxes, "... k d -> (...) k d")

    # draw boxes
    colors = _get_uniform_colors(self.num_colors)
    images = tf.image.draw_bounding_boxes(images, boxes, colors)
    images = np.reshape(images, images_shape)

    # proceed with logic from ShowImages()
    if self.rearrange:
      images = einops.rearrange(images, self.rearrange, **self.rearrange_kwargs)
    if not (len(images.shape) == 4 and images.shape[-1] == 3):
      raise ValueError(f"Bad shape: {images.shape}")

    # maybe rescale
    if self.in_vrange is not None:
      vmin, vmax = self.in_vrange
      images = (images - vmin) / (vmax - vmin)
    # convert to float
    images = media.to_type(images, np.float32)

    # always clip to avoid display problems in TB and Datatables
    images = np.clip(images, 0.0, 1.0)
    # maybe resize
    if (self.width, self.height) != (None, None):
      shape = _get_height_width(self.width, self.height, Shape("h w"))
      images = media.resize_video(images, shape)
    return images


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PerImageChannelPCA(ImageSummary):
  """Reduce the channel dim using PCA to 3dim and show as image."""

  feature_maps: kontext.Key

  num_images: int
  rearrange: Optional[str] = None
  rearrange_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict[str, Any]
  )
  width: Optional[int] = None
  height: Optional[int] = None

  def gather_kwargs(self, context: Any) -> dict[str, Segmentations]:
    # optimize gather_kwargs to only return num_images many images
    kwargs = kontext.resolve_from_keyed_obj(context, self)
    feature_maps = kwargs["feature_maps"]
    if self.rearrange:
      feature_maps = einops.rearrange(
          feature_maps, self.rearrange, **self.rearrange_kwargs
      )
    if not isinstance(feature_maps, Float["*b h w c"]):
      raise ValueError(
          f"Bad shape or dtype: {feature_maps.shape} {feature_maps.dtype}"
      )

    num_images_per_device = math.ceil(
        self.num_images / jax.local_device_count()
    )
    feature_maps = feature_maps[:num_images_per_device]
    return {"feature_maps": feature_maps}

  @typechecked
  def get_images(self, feature_maps: Float["*b h w c"]) -> Float["n _h _w 3"]:
    h, w, _ = feature_maps.shape[-3:]
    # flatten the batch and image dimensions
    features = einops.rearrange(feature_maps, "... h w k -> (...) (h w) k")
    # only use num_images many images and convert to np.arrays
    features = np.array(features[: self.num_images])
    imgs = []
    # Project to top 3 PCA dimensions independently for each example
    for feats in features:
      pca = sklearn.decomposition.PCA(n_components=3)
      feat3 = pca.fit_transform(feats)
      # Normalize to [0, 1]
      vmin = np.min(feat3)
      vmax = np.max(feat3)
      feat3 = (feat3 - vmin) / (vmax - vmin)
      # reshape to image size
      img = einops.rearrange(feat3, "(h w) s -> h w s", h=h, w=w)
      imgs.append(img)
    images = np.array(imgs)

    # maybe resize
    if (self.width, self.height) != (None, None):
      shape = _get_height_width(self.width, self.height, Shape("h w"))
      images = media.resize_video(images, shape)

    images = np.clip(images, 0.0, 1.0)
    return images


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
