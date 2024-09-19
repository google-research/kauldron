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

"""Map transforms."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Optional

import einops
from etils import enp
import flax.core
from kauldron.data.tf import transform_utils
from kauldron.data.transforms import base
from kauldron.typing import TfArray, TfFloat, TfInt, XArray, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import tensorflow as tf

_FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Cast(base.ElementWiseTransform):
  """Cast an element to the specified dtype."""

  dtype: Any

  @typechecked
  def map_element(self, element: TfArray["*any"]) -> TfArray["*any"]:
    return tf.cast(element, self.dtype)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CenterCrop(base.ElementWiseTransform):
  """Crop the input data to the specified shape from the center.

  Can be used on data of any shape or type including images and videos.

  Attributes:
    shape: A tuple of integers describing the target shape of the crop. Entries
      can be also be None to keep the original shape of the data in that dim.
  """

  shape: tuple[Optional[int], ...]

  @typechecked
  def map_element(self, element: TfArray["..."]) -> TfArray["..."]:
    if len(element.shape) != len(self.shape):
      raise ValueError(
          "Rank of self.shape has to match element.shape. But got"
          f" {self.shape=} and {element.shape=}"
      )
    # resolve dynamic portions (-1) of self.shape
    shape = tf.shape(element)
    target_shape = transform_utils.get_target_shape(element, self.shape)
    # compute the offset for the tf.slice
    offset = (shape - target_shape) // 2
    crop = tf.slice(element, offset, target_shape)
    return tf.ensure_shape(crop, self.shape)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Gather(base.ElementWiseTransform):
  """Gathers entries along a single dimension."""

  axis: int
  indices: tuple[int, ...]

  @typechecked
  def map_element(self, element: TfArray) -> TfFloat:
    data = tf.unstack(element, axis=self.axis)
    out = [data[idx] for idx in self.indices]
    return tf.stack(out, axis=self.axis)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class OneHot(base.ElementWiseTransform):
  """One-hot encodes the input.

  Attributes:
    num_classes: Length of the one-hot vector (how many classes).
    multi: If there are multiple labels, whether to merge them into the same
      "multi-hot" vector (True) or keep them as an extra dimension (False).
    on: Value to fill in for the positive label (default: 1).
    off: Value to fill in for negative labels (default: 0).
  """

  num_classes: int
  multi: bool = True
  on: float = 1.0
  off: float = 0.0

  @typechecked
  def map_element(self, labels: TfInt["..."]) -> TfFloat["... M"]:  # pylint: disable=arguments-renamed
    # When there's more than one label, this is significantly more efficient
    # than using tf.one_hot followed by tf.reduce_max; we tested.
    if labels.shape.rank > 0 and self.multi:
      x = tf.scatter_nd(
          labels[:, None], tf.ones(tf.shape(labels)[0]), (self.num_classes,)
      )
      x = tf.clip_by_value(x, 0, 1) * (self.on - self.off) + self.off
    else:
      x = tf.one_hot(
          labels, self.num_classes, on_value=self.on, off_value=self.off
      )

    return x


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Rearrange(base.ElementWiseTransform):
  """Einops rearrange on a single element.

  Mostly a wrapper around einops.rearrange, but also supports basic types like
  int, float, lists and tuples (which are converted to a numpy array first).

  Example:

  ```
  cfg.train_ds = kd.data.Tfds(
      ...
      transforms=[
          ...,
          kd.data.Rearrange(key="image", pattern="h w c -> (h w c)"),
      ]
  )
  ```

  Attributes:
    pattern: `einops.rearrange` pattern, e.g. "b h w c -> b c (h w)"
    axes_lengths: a dictionary for specifying additional axis sizes that cannot
      be inferred from the pattern and the tensor alone.
  """

  pattern: str
  axes_lengths: dict[str, int] = dataclasses.field(default_factory=_FrozenDict)

  @typechecked
  def map_element(self, element: Any) -> XArray:
    # Ensure element is an array (and not a python builtin)
    # This is useful e.g. for pygrain pipelines because often "label" will be
    # int and not an array, yet one might want to reshape it.
    xnp = enp.lazy.get_xnp(element, strict=False)
    element = xnp.asarray(element)

    return einops.rearrange(element, self.pattern, **self.axes_lengths)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Resize(base.ElementWiseTransform):
  """Resize images and corresponding segmentations, etc.

  By default uses resize method "area" for float inputs and resize method
  "nearest" for int inputs.

  Attributes:
    height: Output height of the image(s).
    width: Output width of the image(s).
    method: The resizing method to use. Defaults to "AUTO" in which case the the
      resize method is either "area" (for float inputs) or "nearest" (for int
      inputs). Other possible choices are "bilinear", "lanczos3", "lanczos5",
      "bicubic", "gaussian", "nearest", "area", or "mitchellcubic". See
      `tf.image.resize` for details.
  """

  height: int
  width: int
  method: str = "AUTO"

  @typechecked
  def map_element(self, element: TfArray["*b H W C"]) -> TfArray["*b H2 W2 C"]:
    # Determine resize method based on dtype (e.g. segmentations are int).
    method = self.method
    if method == "AUTO":
      method = "nearest" if element.dtype.is_integer else "area"

    batch_dims = tf.shape(element)[:-3]
    flat_imgs = einops.rearrange(element, "... h w c -> (...) h w c")

    resized_imgs = tf.image.resize(
        flat_imgs, (self.height, self.width), method=method
    )
    return tf.reshape(
        resized_imgs,
        tf.concat([batch_dims, tf.shape(resized_imgs)[-3:]], axis=0),
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ResizeSmall(base.ElementWiseTransform):
  """Resizes the smaller side to `smaller_size` keeping aspect ratio.

  By default uses resize method "area" for float inputs and resize method
  "nearest" for int inputs.

  Attributes:
    smaller_size: an integer, that represents a new size of the smaller side of
      an input image.
    method: The resizing method to use. Defaults to "AUTO" in which case the the
      resize method is either "area" (for float inputs) or "nearest" (for int
      inputs). Other possible choices are "bilinear", "lanczos3", "lanczos5",
      "bicubic", "gaussian", "nearest", "area", or "mitchellcubic". See
      `tf.image.resize` for details.
  """

  smaller_size: int
  method: str = "AUTO"

  @typechecked
  def map_element(self, element: TfArray["*b H W C"]) -> TfArray["*b H2 W2 C"]:
    method = self.method
    if method == "AUTO":
      method = "nearest" if element.dtype.is_integer else "area"

    batch_dims = tf.shape(element)[:-3]

    flat_imgs = einops.rearrange(element, "... h w c -> (...) h w c")

    # Figure out the necessary h/w.
    h = tf.cast(tf.shape(element)[-3], tf.float32)
    w = tf.cast(tf.shape(element)[-2], tf.float32)
    ratio = tf.cast(self.smaller_size, tf.float32) / tf.minimum(h, w)
    h2 = tf.cast(tf.round(ratio * h), tf.int32)
    w2 = tf.cast(tf.round(ratio * w), tf.int32)

    resized_imgs = tf.image.resize(flat_imgs, (h2, w2), method=method)
    return tf.reshape(
        resized_imgs,
        tf.concat([batch_dims, tf.shape(resized_imgs)[-3:]], axis=0),
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ValueRange(base.ElementWiseTransform):
  """Map the value range of an element."""

  vrange: tuple[float, float]
  in_vrange: tuple[float, float] = (0.0, 255.0)

  dtype: Any = tf.float32
  clip_values: bool = True

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    xnp = enp.lazy.get_xnp(element)
    dtype = enp.lazy.as_np_dtype(self.dtype)
    element = xnp.asarray(element, dtype=dtype)
    in_min, in_max = self.in_vrange
    out_min, out_max = self.vrange
    element = (element - in_min) / (in_max - in_min)
    element = element * (out_max - out_min) + out_min
    if self.clip_values:
      element = xnp.clip(element, out_min, out_max)
    return element
