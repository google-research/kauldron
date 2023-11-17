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

"""Preprocessing Ops."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import einops
import flax.core
import grain.tensorflow as grain
from kauldron import kontext

from kauldron.typing import Array, Float, Key, Shape, TfArray, TfFloat, TfInt, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import paths as paths_lib
import numpy as np
import tensorflow as tf


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Elements(grain.MapTransform):
  """Modify the elements by keeping xor dropping and/or renaming."""

  keep: Iterable[str] = ()
  drop: Iterable[str] = ()
  rename: Mapping[str, str] = flax.core.FrozenDict()

  def __post_init__(self):
    if self.keep and self.drop:
      raise ValueError("keep and drop are mutually exclusive")
    keep = set(self.keep)
    drop = set(self.drop)

    rename_keys = set(self.rename.keys())
    rename_values = set(self.rename.values())

    keep_and_rename = keep & (rename_keys | rename_values)
    if keep_and_rename:
      raise KeyError(
          f"Keys: {keep_and_rename} present in both keep and "
          "rename (key or value) collections."
      )

    drop_and_rename = drop & rename_keys
    if drop_and_rename:
      raise KeyError(
          f"Keys: {drop_and_rename} present in both drop and "
          "rename (key) collections."
      )

    drop_rename_meta = grain.META_FEATURES & (
        drop | rename_keys | rename_values
    )
    if drop_rename_meta:
      raise KeyError(
          f"Keys: {drop_rename_meta} are internal keys and may not "
          "be dropped, renamed or overwritten."
      )

    keep_meta = keep & grain.META_FEATURES
    if keep_meta:
      raise KeyError(
          f"Keys: {keep_meta} are internal keys should not be "
          "explicitly kept (happens automatically)."
      )

    object.__setattr__(self, "keep", keep)
    object.__setattr__(self, "drop", drop)

  def map(self, features):
    feature_keys = set(features.keys())
    # resolve keep or drop
    if self.keep:
      keep_keys = set(self.keep)
      missing_keep_keys = keep_keys - feature_keys
      if missing_keep_keys:
        raise KeyError(
            f"keep-key(s) {missing_keep_keys} not found in batch. "
            f"Available keys are {sorted(feature_keys)!r}."
        )
      output = {k: v for k, v in features.items() if k in self.keep}
    elif self.drop:
      drop_keys = set(self.drop)
      missing_drop_keys = drop_keys - feature_keys
      if missing_drop_keys:
        raise KeyError(
            f"drop-key(s) {missing_drop_keys} not found in batch. "
            f"Available keys are {sorted(feature_keys)!r}."
        )
      output = {
          k: v
          for k, v in features.items()
          if k not in self.drop and k not in self.rename
      }
    else:  # only rename
      output = {k: v for k, v in features.items() if k not in self.rename}

    # resolve renaming
    rename_keys = set(self.rename.keys())
    missing_rename_keys = rename_keys - feature_keys
    if missing_rename_keys:
      raise KeyError(
          f"rename-key(s) {missing_rename_keys} not found in batch. "
          f"Available keys are {sorted(feature_keys)!r}."
      )
    renamed = {
        self.rename[k]: v for k, v in features.items() if k in self.rename
    }
    overwrites = sorted(set(renamed.keys()) & set(output.keys()))
    if overwrites:
      offending_renames = [k for k, v in self.rename.items() if v in overwrites]
      raise KeyError(
          f"Tried renaming key(s) {offending_renames!r} to {overwrites!r} but"
          " target names already exist. Implicit overwriting is not supported."
          " Please explicitly drop target keys that should be overwritten."
      )
    output.update(renamed)

    # silently keep grain.META_FEATURES
    output.update(
        {k: v for k, v in features.items() if k in grain.META_FEATURES}
    )
    return output


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class _ElementWise:
  """Mixin class that enables defining a key and iterating relevant elements.

  Mostly used to remove code duplication between ElementWiseTransform and
  ElementWiseRandomTransform.
  """

  key: kontext.Key | Sequence[kontext.Key] | Dict[kontext.Key, kontext.Key]

  def __post_init__(self):
    if not self.key:
      raise KeyError(f"kontext.Key required for {self}")

    # Convert key to Dict[kontext.Key, kontext.Key] format.
    keys = self.key
    if isinstance(self.key, str):
      keys = {self.key: self.key}
    elif isinstance(self.key, Sequence):
      keys = {k: k for k in self.key}
    object.__setattr__(self, "key", keys)

  def _per_element(self, features):
    """Iterator that returns out_key, value pairs."""
    is_noop = True
    for k, v in features.items():
      if k in self.key:
        yield self.key[k], v, True
        is_noop = False
      else:
        yield k, v, False
    if is_noop:
      raise KeyError(
          f"{sorted(self.key.keys())} "  # pytype: disable=attribute-error
          "did not match any keys. Available keys: "
          f"{sorted(features.keys())}"  # pytype: disable=attribute-error
      )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TreeFlattenWithPath(_ElementWise, grain.MapTransform):
  """Flatten any tree-structured elements.

  For example, using 'a' as key, with:
    features = {'a': {'b': 2, 'c': {'d': 3}}, 'e': 5 , 'f': {'g': 6}}

  becomes:
    features = {'a_b': 2, 'a_c_d': 3, 'e': 5, 'f': {'g': 6}}
  """

  separator: str = "_"

  def map(self, features):
    output = {}
    for key, element, should_transform in self._per_element(features):
      if should_transform:
        output.update(
            kontext.flatten_with_path({key: element}, separator=self.separator)
        )
      else:
        output[key] = element
    return output


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseTransform(_ElementWise, grain.MapTransform):
  """Base class for elementwise transforms."""

  def map(self, features):
    return {
        key: self.map_element(element) if should_transform else element
        for key, element, should_transform in self._per_element(features)
    }

  @abc.abstractmethod
  def map_element(self, element):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseRandomTransform(_ElementWise, grain.RandomMapTransform):
  """Base class for elementwise transforms."""

  def random_map(self, features, seed):
    features_out = {}
    for key, element, should_transform in self._per_element(features):
      if should_transform:
        features_out[key] = self.random_map_element(element, seed)
      else:
        features_out[key] = element
    return features_out

  @abc.abstractmethod
  def random_map_element(self, element, seed):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Rearrange(ElementWiseTransform):
  """Einsum rearrange on a single element."""

  pattern: str

  # @typechecked
  def map_element(
      self,
      element: TfArray["..."] | Array["..."],
  ) -> TfArray["..."] | Array["..."]:
    return einops.rearrange(element, self.pattern)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ValueRange(ElementWiseTransform):
  """Map the value range of an element."""

  vrange: tuple[float, float]
  in_vrange: tuple[float, float] = (0.0, 255.0)

  dtype: Any = tf.float32
  clip_values: bool = True

  @functools.singledispatchmethod
  def map_element(self, element):
    pass

  @map_element.register
  def _(self, element: tf.Tensor) -> tf.Tensor:
    in_min_t = tf.constant(self.in_vrange[0], tf.float32)
    in_max_t = tf.constant(self.in_vrange[1], tf.float32)
    vmax = tf.constant(self.vrange[1], tf.float32)
    vmin = tf.constant(self.vrange[0], tf.float32)

    element = tf.cast(element, self.dtype)
    element = (element - in_min_t) / (in_max_t - in_min_t)
    element = element * (vmax - vmin) + vmin
    if self.clip_values:
      element = tf.clip_by_value(element, vmin, vmax)
    return element

  @typechecked
  @map_element.register
  def _(self, element: np.ndarray) -> np.ndarray:
    in_min, in_max = self.in_vrange
    vmin, vmax = self.vrange

    element = (element - in_min) / (in_max - in_min)
    element = element * (vmax - vmin) + vmin

    if self.clip_values:
      element = np.clip(element, a_min=vmin, a_max=vmax)

    NP_DTYPES = {tf.float32: np.float32, np.float32: np.float32}

    return element.astype(dtype=NP_DTYPES[self.dtype])


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class AsNumpy(ElementWiseTransform):
  dtype: Any = None
  broadcast_to: Optional[Shape] = None

  def map_element(self, element: Any) -> np.ndarray:
    np_elem = np.array(element, dtype=self.dtype)
    if self.broadcast_to is not None:
      np_elem = np.broadcast_to(np_elem, self.broadcast_to)
    return np_elem


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandomCrop(ElementWiseRandomTransform):
  """Randomly crop the input data to the specified shape.

  Can be used on data of any shape or type including images and videos.

  Attributes:
    shape: A tuple of integers describing the target shape of the crop. Entries
      can be also be None to keep the original shape of the data in that dim.
  """

  shape: tuple[Optional[int], ...]

  def random_map(self, features, seed):
    if not all([d is None or d >= 0 for d in self.shape]):
      raise ValueError(
          "Target shape can contain only non-negative ints or None. Got"
          f" {self.shape=}"
      )
    shapes = {k: v.shape for k, v in features.items() if k in self.key}
    for key, shape in shapes.items():
      if len(shape) != len(self.shape):
        raise ValueError(
            "Rank of self.shape has to match element shape. But got"
            f" {self.shape=} and {shape=} for {key!r}"
        )
    ref_key, ref_shape = next(iter(shapes.items())) if shapes else (None, None)
    # ensure dimensions match except where self.shape is None
    for key, shape in shapes.items():
      for ref_dim, key_dim, target_dim in zip(ref_shape, shape, self.shape):
        if ref_dim != key_dim and (target_dim is not None):
          raise ValueError(
              "Shapes of different keys for random crop have to be compatible,"
              f" but got {ref_shape} ({ref_key}) != {shape} ({key}) with"
              f" {self.shape=}"
          )

    return super().random_map(features, seed)

  @typechecked
  def random_map_element(self, element: TfArray["..."], seed) -> TfArray["..."]:
    shape = tf.shape(element)
    # resolve dynamic portions of self.shape to a static target_shape
    target_shape = _get_target_shape(element, self.shape)
    # compute the range of the offset for the tf.slice
    offset_range = shape - target_shape
    clipped_offset_range = tf.clip_by_value(offset_range, 1, tf.int32.max)
    # randomly sample offsets from the desired range via modulo
    rand_int = tf.random.stateless_uniform(
        [shape.shape[0]], seed=seed, minval=None, maxval=None, dtype=tf.int32
    )
    offset = tf.where(offset_range > 0, rand_int % clipped_offset_range, 0)
    return tf.slice(element, offset, target_shape)  # crop


def _get_target_shape(t: tf.Tensor, target_shape):
  """Resolve the `dynamic` portions of `target_shape`."""
  finale_shape = []
  dynamic_shape = tf.shape(t)
  for i, (static_dim, target_dim) in enumerate(zip(t.shape, target_shape)):
    if target_dim is not None:
      finale_shape.append(target_dim)
    elif static_dim is not None:
      finale_shape.append(static_dim)
    else:
      finale_shape.append(dynamic_shape[i])
  return finale_shape


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CenterCrop(ElementWiseTransform):
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
    target_shape = tf.constant(  # convert None to -1
        [-1 if s is None else s for s in self.shape], dtype=tf.int32
    )
    target_shape = tf.where(target_shape >= 0, target_shape, shape)
    # compute the offset for the tf.slice
    offset = (shape - target_shape) // 2
    return tf.slice(element, offset, target_shape)  # crop


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class InceptionCrop(ElementWiseRandomTransform):
  """Makes inception-style image crop and optionally resizes afterwards.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Attributes:
    resize_size: Resize image to [resize_size, resize_size] after crop.
    resize_method: The type of interpolation to apply when resizing. Valid
      values those accepted by tf.image.resize.
    area_range: A tuple of (min, max) crop area (as fractions).
  """

  resize_size: Optional[tuple[int, int]] = None
  resize_method: str = tf.image.ResizeMethod.BILINEAR
  area_range: tuple[float, float] = (0.05, 1.0)

  @typechecked
  def random_map_element(  # pylint: disable=arguments-renamed
      self, image: TfArray["*B H W C"], seed
  ) -> TfArray["*B H2 W2 C"]:
    begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
        image_size=tf.shape(image),
        bounding_boxes=tf.zeros([0, 0, 4], tf.float32),
        area_range=self.area_range,
        min_object_covered=0,  # Don't enforce a minimum overlap.
        use_image_if_no_bounding_boxes=True,
        seed=seed,
    )
    crop = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    if self.resize_size is not None:
      crop = tf.image.resize(crop, self.resize_size, self.resize_method)

    return tf.cast(crop, image.dtype)


class RandomFlipLeftRight(ElementWiseRandomTransform):
  """Flips an image horizontally with probability 50%."""

  @typechecked
  def random_map_element(
      self, element: TfArray["*B H W C"], seed
  ) -> TfArray["*B H W C"]:
    return tf.image.stateless_random_flip_left_right(element, seed)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class OneHot(ElementWiseTransform):
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
class Resize(ElementWiseTransform):
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
class ResizeSmall(ElementWiseTransform):
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
class HStack(ElementWiseTransform):
  """Hstack a set of images."""

  @typechecked
  def map_element(  # pylint: disable=arguments-renamed
      self, image: TfFloat["... N H W c"]
  ) -> TfFloat["... H N*W c"]:
    images = tf.unstack(image, axis=-4)
    return tf.concat(images, axis=-2)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class VStack(ElementWiseTransform):
  """Vstack a set of images."""

  @typechecked
  def map_element(  # pylint: disable=arguments-renamed
      self, image: TfFloat["... N H W c"]
  ) -> TfFloat["... N*H W c"]:
    images = tf.unstack(image, axis=-4)
    return tf.concat(images, axis=-3)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PadFirstDimensionToFixedSize(ElementWiseTransform):
  """Pads 0-axis of a tensor to a fixed size (can be used for sparse -> dense)."""

  size: int
  value: int = -1

  @typechecked
  def map_element(self, element):
    pad_size = tf.maximum(self.size - tf.shape(element)[0], 0)
    padded_element = tf.pad(
        element,
        ((0, pad_size),) + ((0, 0),) * (len(element.shape) - 1),  # pytype: disable=attribute-error  # allow-recursive-types
        constant_values=self.value,
    )

    return tf.ensure_shape(
        padded_element,
        [self.size]
        + (
            element.get_shape().as_list()[1:]
            if len(tf.shape(element)) > 1
            else []
        ),
    )
