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

"""Map transforms."""

from collections.abc import Iterable
import dataclasses
import typing
from typing import Any

import einops
from etils import enp
from etils import epy
import flax.core
import jax
from kauldron.data.transforms import base
from kauldron.typing import Shape, XArray, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np

with epy.lazy_imports():
  import tensorflow as tf  # pylint: disable=g-import-not-at-top

_FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Cast(base.ElementWiseTransform):
  """Cast an element to the specified dtype."""

  dtype: Any

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    if enp.lazy.is_tf(element):
      return tf.cast(element, self.dtype)
    else:
      return element.astype(self.dtype)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Rearrange(base.ElementWiseTransform):
  """Einops rearrange on a single element.

  Mostly a wrapper around einops.rearrange, but also supports basic types like
  int, float, lists and tuples (which are converted to a numpy array first).

  Example:

  ```
  cfg.train_ds = kd.data.tf.Tfds(
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
class ValueRange(base.ElementWiseTransform):
  """Map the value range of an element."""

  vrange: tuple[float, float]
  in_vrange: tuple[float, float] = (0.0, 255.0)

  dtype: Any = np.float32
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
    # Intervening operations may have changed the dtype, so cast back.
    element = xnp.asarray(element, dtype=dtype)
    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Gather(base.ElementWiseTransform):
  """Gathers entries along a single dimension."""

  axis: int
  indices: tuple[int, ...]

  @typechecked
  def map_element(self, element: XArray) -> XArray:
    xnp = enp.lazy.get_xnp(element)
    return xnp.take(element, self.indices, axis=self.axis)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Resize(base.ElementWiseTransform):
  """Resizes an image.

  At most one of `size`, `min_size`, and `max_size` must be set.

  Attributes:
    size: The new size of the image. If set, the image is rescaled so that the
      new size matches `size`.
    min_size: The minimum size of the image. If set, the image is rescaled so
      that the smaller edge matches `min_size`.
    max_size: The maximum size of the image. If set, the image is rescaled so
      that the larger edge matches `max_size`.
    method: The resizing method. If `None`, uses `area` for float TF inputs,
      `bilinear` for float JAX inputs, and `nearest` for int inputs.
    antialias: Whether to use an anti-aliasing filter.
  """

  size: tuple[int, int] | None = None
  min_size: int | None = None
  max_size: int | None = None
  method: str | jax.image.ResizeMethod | tf.image.ResizeMethod | None = None
  antialias: bool = True

  def __post_init__(self):
    super().__post_init__()
    self._validate_params()

  @typechecked
  def map_element(self, element: XArray["*b h w c"]) -> XArray["*b h2 w2 c"]:
    if self.method is None:
      if _is_integer(element.dtype):
        method = "nearest"
      elif enp.lazy.is_tf(element):
        method = "area"
      else:  # Jax or Np
        method = "bilinear"
    else:
      method = self.method

    new_size = self._get_resize_shape(*Shape("h w"))
    if enp.lazy.is_tf(element):
      # Flatten the batch dimensions
      batch = tf.shape(element)[:-3]
      imgs = einops.rearrange(element, "... h w c -> (...) h w c")

      imgs = tf.image.resize(
          imgs,
          new_size,
          method=method,
          antialias=self.antialias,
      )

      # Unflatten the batch dimensions
      return tf.reshape(imgs, tf.concat([batch, tf.shape(imgs)[-3:]], axis=0))
    elif enp.lazy.is_np(element) or enp.lazy.is_jax(element):
      if method == "area":
        raise ValueError(
            "Area resizing is not supported in JAX for float inputs"
            " (Upvote: https://github.com/jax-ml/jax/issues/20098).\n"
            "Please explicitly provide a resizing method."
        )

      *batch, _, _, c = element.shape
      size = (*batch, *new_size, c)
      # Explicitly set device to avoid `Disallowed host-to-device transfer`
      # Uses default sharding.
      element = jax.device_put(element, jax.local_devices(backend="cpu")[0])
      return jax.image.resize(
          element,
          size,
          method=method,
          antialias=self.antialias,
      )
    else:
      raise ValueError(f"Unsupported type: {type(element)}")

  def _validate_params(self) -> None:
    number_sizes_set = sum([
        self.size is not None,
        self.min_size is not None,
        self.max_size is not None,
    ])
    if number_sizes_set != 1:
      raise ValueError(
          "Exactly one of `size`, `min_size`, and `max_size` must be set. "
          f"Got {number_sizes_set} sizes set."
      )

  def _get_resize_shape(self, h: int, w: int) -> tuple[int, int]:
    if self.size is not None:
      return self.size
    elif self.min_size is not None:
      ratio = self.min_size / min(h, w)
      return int(round(h * ratio)), int(round(w * ratio))
    elif self.max_size is not None:
      ratio = self.max_size / max(h, w)
      return int(round(h * ratio)), int(round(w * ratio))
    else:
      raise ValueError("One of `size`, `min_size`, and `max_size` must be set.")


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CenterCrop(base.ElementWiseTransform):
  """Crop the input data to the specified shape from the center.

  Can be used on data of any shape or type including images and videos. This
  transform does NOT support dynamic shapes, i.e., shapes with -1 in any of
  the dimensions.

  Attributes:
    shape: A tuple of integers describing the target shape of the crop. Entries
      can be also be None to keep the original shape of the data in that dim.
  """

  shape: tuple[int | None, ...]

  @typechecked
  def map_element(self, element: XArray["..."]) -> XArray["..."]:
    if len(element.shape) != len(self.shape):
      raise ValueError(
          "Rank of self.shape has to match element.shape. But got"
          f" {self.shape=} and {element.shape=}"
      )
    target_shape = self._resolve_target_shape(element.shape, self.shape)
    start = np.subtract(element.shape, target_shape) // 2
    if enp.lazy.is_tf(element):
      crop = tf.slice(element, start, target_shape)
      return tf.ensure_shape(crop, target_shape)
    elif enp.lazy.is_np(element) or enp.lazy.is_jax(element):
      end = np.add(start, target_shape)
      return jax.lax.slice(element, start, end)
    else:
      raise ValueError(f"Unsupported type: {type(element)}")

  def _resolve_target_shape(
      self, element_shape: Iterable[int], crop_shape: tuple[int | None, ...]
  ) -> tuple[int, ...]:
    final_shape = []
    for e, c in zip(element_shape, crop_shape):
      final_shape.append(c or e)
    return tuple(final_shape)


def _is_integer(dtype: Any) -> bool:
  return np.issubdtype(enp.lazy.as_dtype(dtype), np.integer)
