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

"""Preprocessing ops."""

from __future__ import annotations

import abc
from collections.abc import Mapping
import dataclasses
import typing
from typing import Any, Callable, List, Literal, Optional, Sequence

import einops
from etils import enp
import flax.core
import grain.tensorflow as grain
import jax
from jax import tree_util
from kauldron import kd
from kauldron.typing import PyTree, TfArray, TfFloat, XArray, typechecked  # pylint: disable=g-importing-member,g-multiple-import
import numpy as np
import tensorflow as tf


# Alias of `third_party/py/jax/_src/tree_util.py:KeyPath` which is not included
# in public API.
KeyEntry = Any
KeyPath = tuple[KeyEntry, ...]

FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Binarize(kd.data.ElementWiseTransform):
  """Binarizes a float tensor by threshold."""

  threshold: float

  @typechecked
  def map_element(self, element: TfArray["*any"]) -> TfArray["*any"]:
    return tf.cast(element > self.threshold, dtype=tf.int32)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CreateMask(kd.data.MapTransform):
  """Creates mask tensor by checking a special mask value in original tensor.

  Variants:
  mask_key: instead of masking based on key directly, can do it based on a
      different mask_key.
  target_key: instead of writing mask result in key + "_mask", can write it in
      target_key + "_mask"
  Dimensions of output are same as input, but channel dimension is 1
  """

  key: str
  mask_key: str | None = None
  target_key: str | None = None

  def map(self, batch):
    # create mask tensor
    if self.mask_key is None:
      mask_key = self.key
    else:
      mask_key = self.mask_key

    mask_value = batch[mask_key]
    if enp.lazy.is_tf(mask_value):
      mask_value = tf.cast(mask_value, dtype=tf.float32)
      mask_tensor = self.condition(mask_value)
      mask_tensor = tf.cast(mask_tensor, dtype=tf.float32)
    elif enp.lazy.is_np(mask_value) or enp.lazy.is_jax(mask_value):
      xnp = enp.lazy.get_xnp(mask_value, strict=False)
      mask_value = mask_value.astype(xnp.float32)
      mask_tensor = self.condition(mask_value)
      mask_tensor = mask_tensor.astype(xnp.float32)
    else:
      raise ValueError(
          f"Unsupported type for mask_key: {type(batch[mask_key])}"
      )
    # return the mask itself
    target_key = self.key
    if self.target_key is not None:
      target_key = self.target_key

    batch[target_key + "_mask"] = mask_tensor
    return batch

  @abc.abstractmethod
  def condition(self, tensor):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CreateMeshGridMask(CreateMask):
  """Creates mesh grid mask to be used with cropping."""

  def condition(self, tensor):
    if enp.lazy.is_tf(tensor):
      shape = tf.shape(tensor)
      height, width = shape[-3], shape[-2]
      x, y = tf.meshgrid(tf.range(width), tf.range(height))
      return tf.stack((x, y), axis=-1)[None]
    elif enp.lazy.is_np(tensor) or enp.lazy.is_jax(tensor):
      xnp = enp.lazy.get_xnp(tensor, strict=False)
      height, width = tensor.shape[-3], tensor.shape[-2]
      x, y = xnp.meshgrid(xnp.arange(width), xnp.arange(height))
      return xnp.stack((x, y), axis=-1)[None]
    else:
      raise ValueError(f"Unsupported type for mask_key: {type(tensor)}")


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CreateRangeMask(CreateMask):
  """Creates mask tensor to mask out values outside a range."""

  min_value: float
  max_value: float

  def condition(self, tensor):
    return (tensor >= self.min_value) & (tensor <= self.max_value)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CreateNonEqualMask(CreateMask):
  """Creates mask tensor to mask out values that are not equal to a value."""

  mask_value: float

  def condition(self, tensor):
    xnp = enp.lazy.get_xnp(tensor, strict=False)
    return xnp.equal(tensor, self.mask_value)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ExtractInitialFixedLengthClip(kd.data.ElementWiseTransform):
  """Extracts a fixed-length clip from the beginning of a sequence of images."""

  num_frames: int

  @typechecked
  def map_element(self, sequence: List[XArray]) -> XArray:

    if self.num_frames <= 0:
      raise ValueError("Number of frames must be positive.")

    if not sequence:
      raise ValueError("Sequence must not be empty.")

    first_clip = sequence[: self.num_frames]

    xnp = enp.lazy.get_xnp(first_clip[0], strict=False)

    # Pad if needed
    padding_len = max(0, self.num_frames - len(sequence))
    if padding_len:
      padding_frame = xnp.zeros_like(sequence[0])
      first_clip = first_clip + [padding_frame] * padding_len

    first_clip = xnp.stack(first_clip)

    return first_clip


@dataclasses.dataclass(frozen=True, eq=True)
class SliceVideosIntoFrames(grain.UnsafeTfDataTransform):
  """Slices sequence features into separate tensors.

  In addition, adds context tensors and grain features to the each resulting
  slice.
  """

  tensors_to_slice_names: Sequence[str]
  context_tensors_names: Sequence[str]

  def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:

    def slice_and_replicate(batch):
      assert (
          self.tensors_to_slice_names
      ), "The list of tensors to slie cannot be empty."
      slicable_batch = {}
      for key in self.tensors_to_slice_names:
        slicable_batch[key] = batch[key]
        # All slicable tensors must have the same 0th dimension.
      num_slices = batch[self.tensors_to_slice_names[0]].shape[0]

      context_batch = {}
      for key in self.context_tensors_names:
        context_batch[key] = batch[key]
      grain_batch = {}
      for key in grain.META_FEATURES:
        if key in batch:
          grain_batch[key] = batch[key]
      if self.context_tensors_names:
        return tf.data.Dataset.zip(
            tf.data.Dataset.from_tensor_slices(slicable_batch),
            tf.data.Dataset.from_tensors(context_batch).repeat(num_slices),
            tf.data.Dataset.from_tensors(grain_batch).repeat(num_slices),
        )
      else:
        return tf.data.Dataset.zip(
            tf.data.Dataset.from_tensor_slices(slicable_batch),
            tf.data.Dataset.from_tensors(grain_batch).repeat(num_slices),
        )

    def to_dict(*arg):
      result = {}
      for t in arg:
        result.update(t)
      return result

    return dataset.flat_map(slice_and_replicate).map(to_dict)


@dataclasses.dataclass(frozen=True, eq=True)
class AddConstants(grain.MapTransform):
  """Adds constant elements for missing fields, eg. as needed for mixtures."""

  values: Mapping[str, Any] = flax.core.FrozenDict()

  def map(self, features):
    overwrites = sorted(set(self.values.keys()) & set(features.keys()))
    if overwrites:
      offending_keys = [k for k, v in self.values.items() if v in overwrites]
      raise KeyError(
          f"Tried adding key(s) {offending_keys!r} but"
          " target names already exist. Implicit overwriting is not supported."
          " Please explicitly drop target keys that should be overwritten."
      )
    features.update(self.values)
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandomResize(kd.data.tf.ElementWiseRandomTransform):
  """Scales video randomly between a set of factors.

  """

  prob: float = 0.8
  min_scale_factor: float = 0.8
  max_scale_factor: float = 1.2
  method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR
  antialias: bool = False

  @typechecked
  def random_map_element(
      self,
      element: TfArray["*B H W C"],
      seed,
  ) -> TfArray["*B h w C"]:

    h_scale = tf.random.stateless_uniform(
        shape=[],
        seed=seed,
        minval=self.min_scale_factor,
        maxval=self.max_scale_factor,
    )
    w_scale = tf.random.stateless_uniform(
        shape=[],
        seed=seed,
        minval=self.min_scale_factor,
        maxval=self.max_scale_factor,
    )
    h = tf.cast(tf.shape(element)[-3], tf.float32)
    w = tf.cast(tf.shape(element)[-2], tf.float32)
    resize_height = tf.cast(h_scale * h, tf.int32)
    resize_width = tf.cast(w_scale * w, tf.int32)
    resized_element = tf.image.resize(
        element,
        (resize_height, resize_width),
        method=self.method,
        antialias=self.antialias,
    )
    coin_toss = tf.random.stateless_uniform(
        (), minval=0, maxval=1, dtype=tf.float32, seed=seed
    )
    element = tf.cond(
        pred=tf.less(coin_toss, tf.cast(self.prob, tf.float32)),
        true_fn=lambda: resized_element,
        false_fn=lambda: tf.cast(element, tf.float32),
    )
    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GaussianBlur(kd.data.tf.ElementWiseRandomTransform):
  """Gaussian Blur.

  """

  apply_prob: float = 0.5
  sigma_min: float = 0.1
  sigma_max: float = 2.0
  kernel_size: int = 9

  @typechecked
  def random_map_element(
      self,
      element: TfArray["*B H W C"],
      seed,
  ) -> TfArray["*B h w C"]:

    # Randomly sample a sigma value.
    # Sigma corresponds to the standard deviation of the Gaussian kernel.
    sigma = tf.random.stateless_uniform(
        [], seed, minval=self.sigma_min, maxval=self.sigma_max, dtype=tf.float32
    )

    # Converts kernel size into odd integer to ensure center pixel.
    kernel_size = 2 * int(self.kernel_size / 2) + 1

    # Creates a 1D kernel of that size and sets it to be a Gaussian.
    x = tf.cast(tf.range(-(kernel_size // 2), kernel_size // 2 + 1), tf.float32)
    blur_filter = tf.exp(-(x**2) / (2.0 * sigma**2))
    # Normalizes the kernel to sum to 1.
    blur_filter = blur_filter / tf.reduce_sum(blur_filter)

    # Creates 1D filters horizontally and vertically to achieve 2D
    # convolution. This works because the Gaussian is separable.
    blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
    blur_v = tf.tile(blur_v, [1, 1, element.shape[-1], 1])
    blur_h = tf.transpose(blur_v, [1, 0, 2, 3])

    # Does the actual blurring using depthwise_conv2d.
    blurred = tf.nn.depthwise_conv2d(
        element, blur_h, strides=[1, 1, 1, 1], padding="SAME"
    )
    blurred = tf.nn.depthwise_conv2d(
        blurred, blur_v, strides=[1, 1, 1, 1], padding="SAME"
    )

    # Randomly apply the blur based on apply_prob.
    coin_toss = tf.random.stateless_uniform(
        (), minval=0, maxval=1, dtype=tf.float32, seed=seed
    )
    element = tf.cond(
        pred=tf.less(coin_toss, tf.cast(self.apply_prob, tf.float32)),
        true_fn=lambda: blurred,
        false_fn=lambda: element,
    )
    return element


class RandomFlipLeftRightVideo(kd.data.tf.ElementWiseRandomTransform):
  """Randomly flips an all frames in input video.

  For an input of shape (B,H,W,C), this transformation randomly
  flips all elements in batch B horizontally with 50% probability
  of being flipped.

  This differs from kd.data.tf.RandomFlipLeftRight which flips every element in
  B with an independent probability of 50%, here there is one 50% probability
  that applies to all elements in B. This means the video is consistently
  flipped.
  """

  @typechecked
  def random_map_element(
      self, element: TfArray["B H W C"], seed
  ) -> TfArray["B H W C"]:
    flip_mask = tf.random.stateless_uniform(shape=(1, 1, 1, 1), seed=seed) < 0.5

    # Apply the mask to the batch
    flipped_images = tf.where(
        flip_mask, tf.image.flip_left_right(element), element
    )
    return flipped_images


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class OneMinus(kd.data.ElementWiseTransform):
  """One minus an element (e.g. for inverting a mask)."""

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    return 1 - element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class HStack(kd.data.ElementWiseTransform):
  """Hstack a set of images."""

  @typechecked
  def map_element(  # pylint: disable=arguments-renamed
      self, image: TfFloat["... N H W c"]
  ) -> TfFloat["... H N*W c"]:
    images = tf.unstack(image, axis=-4)
    return tf.concat(images, axis=-2)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class JaxImageResize(kd.data.ElementWiseTransform):
  """Resizes an element using jax.image.resize."""

  shape: tuple[int, ...]
  method: str = (
      # can be "nearest", "linear", "bilinear", "trilinear", "triangle",
      # "cubic", "bicubic", "tricubic", "lanczos3", "lanczos5"
      "bilinear"
  )

  def map_element(self, element: XArray) -> XArray:
    return jax.image.resize(element, shape=self.shape, method=self.method)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class VStack(kd.data.ElementWiseTransform):
  """Vstack a set of images."""

  @typechecked
  def map_element(  # pylint: disable=arguments-renamed
      self, image: TfFloat["... N H W c"]
  ) -> TfFloat["... N*H W c"]:
    images = tf.unstack(image, axis=-4)
    return tf.concat(images, axis=-3)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PadAxisToFixedSize(kd.data.ElementWiseTransform):
  """Pads axis of a tensor to a fixed size (can be used for sparse -> dense)."""

  size: int
  axis: int
  value: int = -1

  @typechecked
  def map_element(self, element):
    pad_size = tf.maximum(self.size - tf.shape(element)[self.axis], 0)

    padding = (
        [(0, 0)] * len(element.shape[: self.axis])
        + [(0, pad_size)]
        + [(0, 0)] * len(element.shape[self.axis + 1 :])
    )
    padded_element = tf.pad(
        element,
        padding,
        constant_values=self.value,
    )

    new_shape = (
        element.get_shape().as_list()[: self.axis]
        + [self.size]
        + element.get_shape().as_list()[self.axis + 1 :]
    )
    return tf.ensure_shape(padded_element, new_shape)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Repeat(kd.data.ElementWiseTransform):
  """Einops repeat on a single element.

  Mostly a wrapper around einops.repeat, but also supports basic types like
  int, float, lists and tuples (which are converted to a numpy array first).

  Example:

  ```
  cfg.train_ds = kd.data.tf.Tfds(
      ...
      transforms=[
          ...,
          kd.data.Repeat(key="image", pattern="h w c -> t h w c",
                         axes_lengths={"t": 6}),
      ]
  )
  ```

  Attributes:
    pattern: `einops.repeat` pattern, e.g. "b h w c -> b c (h w)"
    axes_lengths: a dictionary for specifying additional axis e.g. number of
      repeats or axis that cannot be inferred from the pattern and the tensor
      alone.
  """

  pattern: str
  axes_lengths: dict[str, int] = dataclasses.field(default_factory=FrozenDict)

  @typechecked
  def map_element(self, element: Any) -> XArray:
    # Ensure element is an array (and not a python builtin)
    # This is useful e.g. for pygrain pipelines because often "label" will be
    # int and not an array, yet one might want to reshape it.
    xnp = enp.lazy.get_xnp(element, strict=False)
    element = xnp.asarray(element)

    return einops.repeat(element, self.pattern, **self.axes_lengths)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TemporalRandomWindow(kd.data.tf.ElementWiseRandomTransform):
  """Gets a random slice (window) along 0-th axis of input tensor.

  Pads the input tensor along the time axis if the tensor length is shorter than
  the provided length. Supports padding with the last value or a constant value.

  Attr:
    length: An integer representing the desired length of the output tensor.
    padding_mode: Either "last" or "constant", specifying the padding strategy.
      "last" repeats the last value of the input tensor. "constant" pads with a
      constant value.
    padding_value: A float defining the value with which to pad when
      `padding_mode` is "constant".  Ignored if `padding_mode` is "last".
    frame_rate: The frame rate to use if `random_frame_rate` is False.
  """

  length: int
  padding_mode: Literal["constant", "last"] = "constant"
  padding_value: float = 0.0
  frame_rate: int | Literal["random"] = 1

  @typechecked
  def random_map_element(  # pylint: disable=arguments-renamed
      self, tensor: TfArray["T *C"], seed
  ) -> TfArray["t *C"]:
    length = tf.minimum(self.length, tf.shape(tensor)[0])

    rank = len(tensor.shape)
    if self.frame_rate == "random":
      max_frame_rate = tf.cast(tf.floor(tf.shape(tensor)[0] / length), tf.int32)
      frame_rate = tf.random.stateless_uniform(
          shape=[],
          seed=seed,
          minval=1,
          maxval=max_frame_rate + 1,
          dtype=tf.int32,
      )
    else:
      frame_rate = self.frame_rate
    length = frame_rate * length
    window_size = tf.concat(([length], tf.shape(tensor)[1:]), axis=0)
    tensor = tf.image.stateless_random_crop(tensor, size=window_size, seed=seed)
    indices = tf.range(start=0, limit=tf.shape(tensor)[0], delta=frame_rate)
    tensor = tf.gather(tensor, indices)
    frames_to_pad = tf.maximum(self.length - tf.shape(tensor)[0], 0)

    if self.padding_mode == "constant":
      tensor = tf.pad(
          tensor,
          ((0, frames_to_pad),) + ((0, 0),) * (rank - 1),
          constant_values=self.padding_value,
      )
    elif self.padding_mode == "last":
      padding = tf.tile(tensor[-1:], [frames_to_pad] + [1] * (rank - 1))
      tensor = tf.concat([tensor, padding], axis=0)
    else:
      raise ValueError(f"Unknown padding mode: {self.padding_mode}")
    tensor = tf.ensure_shape(tensor, [self.length] + tensor.get_shape()[1:])
    return tf.cast(tensor, tensor.dtype)


# TODO(glemoing): factor out all the TemporalXXX transforms in temporal.py file
# TODO(glemoing): add random frame rate option
@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TemporalRandomWalk(kd.data.tf.ElementWiseRandomTransform):
  """Performs a random walk along 0-th axis of input tensor.

  Starts at a random index, moves a random number of steps in a random
  direction (forward or backward), then changes direction and repeats until
  the desired length is reached. Handles boundary conditions to prevent going
  out of bounds.

  Attr:
    length: An integer representing the new length of the video.
    thresh: A float between 0 and 1 defining the probability of changing
      direction at each step.
  """

  length: int
  thresh: float = 0.05

  @typechecked
  def random_map_element(  # pylint: disable=arguments-renamed
      self, tensor: TfArray["T *C"], seed
  ) -> TfArray["t *C"]:
    tensor = tf.concat([tensor, tf.reverse(tensor[1:-1], axis=[0])], axis=0)
    input_length = tf.shape(tensor)[0]
    start_step = tf.random.stateless_uniform(
        shape=[], minval=0, maxval=input_length, dtype=tf.int32, seed=seed
    )
    change_of_direction = tf.random.stateless_uniform(
        shape=[self.length], minval=0.0, maxval=1.0, seed=seed
    )
    change_of_direction = tf.cast(change_of_direction < self.thresh, tf.int32)
    direction = 1 - 2 * change_of_direction
    direction = tf.math.cumprod(direction, axis=0)
    steps = start_step + tf.cumsum(direction, axis=0)
    steps = tf.math.floormod(steps, input_length)
    tensor = tf.gather(tensor, steps, axis=0)
    return tensor


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class SliceWithStride(kd.data.ElementWiseTransform):
  """Slices a tensor with stride and keeps the specified number of elements.

  Attr:
    stride: An int defining the stride.
    num_elements: An int defining the number of elements to keep.
  """

  stride: int
  num_elements: Optional[int] = None

  @typechecked
  def map_element(  # pylint: disable=arguments-renamed
      self, tensor: XArray["T *C"]
  ) -> XArray["t *C"]:
    tensor = tensor[:: self.stride]
    if self.num_elements is not None:
      tensor = tensor[: self.num_elements]
    return tensor


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TemporalRandomStridedWindow(kd.data.tf.ElementWiseRandomTransform):
  """Gets a random strided slice (window) along 0-th axis of input tensor.

  Pads the input tensor along the time axis if the tensor length is shorter than
    the provided length.

  For instance, say we have the following video:
    [1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]

  If length is 4 and stride is 3, this op will randomly sample one of the
  following windows:
    [1, 0, 0, 2]
    [2, 0, 0, 3]
    [3, 0, 0, 4]
    [4, 0, 0, 0]  # with padding

  Attr:
    length: An integer representing the new length of the video.
    padding_value: A float defining the value with which we pad.
  """

  length: int
  stride: int
  padding_value: float = 0.0

  @typechecked
  def random_map_element(  # pylint: disable=arguments-renamed
      self, tensor: TfArray["T *C"], seed
  ) -> TfArray["t *C"]:
    all_frames = tf.signal.frame(
        tensor,
        frame_length=self.length,
        frame_step=self.stride,
        axis=0,
        pad_end=True,
        pad_value=self.padding_value,
    )
    random_index = tf.random.stateless_uniform(
        shape=(),
        seed=seed,
        minval=0,
        maxval=tf.shape(all_frames)[0],
        dtype=tf.int32,
    )
    tensor = all_frames[random_index]

    tensor = tf.ensure_shape(tensor, [self.length] + tensor.get_shape()[1:])
    return tf.cast(tensor, tensor.dtype)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PadImage(kd.data.ElementWiseTransform):
  """Pad image."""

  pad: int
  mode: str

  @typechecked
  def map_element(self, element: TfArray["*b H W C"]) -> TfArray["*b H2 W2 C"]:
    batch_dims = len(element.shape[:-3])
    padding = ((0, 0),) * batch_dims + (
        (self.pad, self.pad),
        (self.pad, self.pad),
        (0, 0),
    )
    return tf.pad(element, padding, mode=self.mode)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PadImageEdgeVal(kd.data.ElementWiseTransform):
  """Pad image using edge values.

  tf.pad has no edge mode, so use SYMMETRIC mode n times to pad n pixels.
  This approach will be slow for large padding, creating a new matrix and
  copying values across may then be preferred.
  """

  pad: int

  @typechecked
  def map_element(self, element: TfArray["*b H W C"]) -> TfArray["*b H2 W2 C"]:
    batch_dims = len(element.shape[:-3])
    padding = ((0, 0),) * batch_dims + ((1, 1), (1, 1), (0, 0))

    for _ in range(self.pad):
      element = tf.pad(element, padding, mode="SYMMETRIC")
    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandomSubsetAlongAxis(kd.data.tf.ElementWiseRandomTransform):
  """Random take a subset of elements along specified axis.

  Note that it current uses the same shuffle subset for all batches.

  Attributes:
    num_to_keep:  Number of positions to keep along the axis.  They are in a
      random order.
    axis:  Which axis to take the subset along.
  """

  num_to_keep: int
  axis: int

  @typechecked
  def random_map_element(self, element: TfArray["..."], seed) -> TfArray["..."]:
    indices = tf.random.experimental.index_shuffle(
        index=tf.range(self.num_to_keep),
        seed=seed,
        max_index=tf.shape(element)[self.axis] - 1,  # Range is inclusive!
    )
    shuffled_vector = tf.gather(element, indices, axis=self.axis)
    return shuffled_vector


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class AddBias(kd.data.ElementWiseTransform):
  """Adds a constant scalar value to the chosen arrays."""

  bias: float

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    return element + self.bias


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Scale(kd.data.ElementWiseTransform):
  """Scale an element by multiplying by a factor."""

  factor: float

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    return element * self.factor


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Standardize(kd.data.ElementWiseTransform):
  """Standardizes an element by computing (x - mu) / std."""

  mean: XArray["N"]
  std: XArray["N"]

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    return (element - self.mean) / self.std


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ApplyW(kd.data.ElementWiseTransform):
  """Applies a matrix projection."""

  w: TfArray["N N"]

  @typechecked
  def map_element(self, element: TfArray["*any"]) -> TfArray["*any"]:
    return tf.squeeze(tf.matmul(self.w, tf.expand_dims(element, -1)), -1)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseTransformWithPredicate(grain.MapTransform):
  """Base class for elementwise transforms.

  This class is intended to allow key and/or feature-dependent logic to
  determine whether to apply a particular transformation, e.g. running a
  rescaling operation on keys containing 'rgb' or 'image'.
  """

  # A predicate accepting key (or KeyPath, if `features` is a PyTree) and
  # value inputs and returning whether to transform that feature.
  should_transform_pred: Callable[[str | KeyPath, Any], bool] = (
      lambda k, v: False
  )

  def map(self, features: PyTree[Any]) -> PyTree[Any]:
    def maybe_transform(path: KeyPath, feature: Any):
      should_transform = self.should_transform_pred(path, feature)
      return self.map_element(feature) if should_transform else feature

    return tree_util.tree_map_with_path(maybe_transform, features)

  @abc.abstractmethod
  def map_element(self, element):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseRandomTransformWithPredicate(grain.RandomMapTransform):
  """Base class for elementwise random transforms.

  This class is intended to allow key and/or feature-dependent logic to
  determine whether to apply a particular transformation, e.g. running a
  rescaling operation on keys containing 'rgb' or 'image'.
  """

  # A predicate accepting key (or KeyPath, if `features` is a PyTree) and
  # value inputs and returning whether to transform that feature.
  should_transform_pred: Callable[[str | KeyPath, Any], bool] = (
      lambda k, v: False
  )

  def random_map(self, features: PyTree[Any], seed) -> PyTree[Any]:
    def maybe_transform(path: KeyPath, feature: Any):
      should_transform = self.should_transform_pred(path, feature)
      return (
          self.random_map_element(feature, seed)
          if should_transform
          else feature
      )

    return tree_util.tree_map_with_path(maybe_transform, features)

  @abc.abstractmethod
  def random_map_element(self, element, seed):
    raise NotImplementedError


class FlipUpsideDown(ElementWiseTransformWithPredicate):
  """Flips an image vertically (upside down)."""

  @typechecked
  def map_element(self, element: TfArray["*B H W C"]) -> TfArray["*B H W C"]:
    return tf.reverse(element, axis=[-3])

  @classmethod
  def matching_keys(
      cls,
      strs: str | Sequence[str] = ("img", "image", "rgb"),
  ):
    """Builds a `ValueRange` matching specific strings against the keys."""
    strs = [strs] if isinstance(strs, str) else strs

    def should_transform_pred(k: str | KeyPath, _):
      if isinstance(k, tuple) and isinstance(k[-1], tree_util.DictKey):
        leaf_key = k[-1].key
      elif isinstance(k, str):
        leaf_key = k
      else:
        raise ValueError(f"Unsupported key type: {type(k)}")

      matched = any([s in leaf_key for s in strs])
      return matched

    return cls(should_transform_pred=should_transform_pred)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ValueRange(ElementWiseTransformWithPredicate):
  """Map the value range of an element.

  This is a fork of `kd.data.ValueRange` but using
  ElementWiseTransformWithPredicate rather than ElementWiseTransform to allow
  more flexible application.
  """

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

  @classmethod
  def matching_keys(
      cls,
      *,
      vrange: tuple[float, float],
      in_vrange: tuple[float, float] = (0.0, 255.0),
      dtype: Any = tf.float32,
      clip_values: bool = True,
      strs: str | Sequence[str] = ("img", "image", "rgb"),
      strs_to_exclude: str | Sequence[str] = (),
  ):
    """Builds a `ValueRange` matching specific strings against the keys."""
    strs = [strs] if isinstance(strs, str) else strs
    strs_to_exclude = (
        [strs_to_exclude]
        if isinstance(strs_to_exclude, str)
        else strs_to_exclude
    )

    def should_transform_pred(k: str | KeyPath, _):
      if isinstance(k, tuple) and isinstance(k[-1], tree_util.DictKey):
        leaf_key = k[-1].key
      elif isinstance(k, str):
        leaf_key = k
      else:
        raise ValueError(f"Unsupported key type: {type(k)}")

      matched = any([s in leaf_key for s in strs])
      to_exclude = any([s in leaf_key for s in strs_to_exclude])
      return matched and (not to_exclude)

    return cls(
        vrange=vrange,
        in_vrange=in_vrange,
        dtype=dtype,
        clip_values=clip_values,
        should_transform_pred=should_transform_pred,
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Resize(ElementWiseTransformWithPredicate):
  """Resize images and corresponding segmentations, etc.

  By default uses resize method "area" for float inputs and resize method
  "nearest" for int inputs.

  This is a fork of `kd.data.tf.Resize` but using
  ElementWiseTransformWithPredicate rather than ElementWiseTransform to allow
  more flexible application.

  Attributes:
    height: Output height of the image(s).
    width: Output width of the image(s).
    method: The resizing method to use. Defaults to "AUTO" in which case the
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

  @classmethod
  def matching_keys(
      cls,
      *,
      height: int,
      width: int,
      method: str = "AUTO",
      strs: str | Sequence[str] = ("img", "image", "rgb"),
      strs_to_exclude: str | Sequence[str] = (),
  ):
    """Builds a `ValueRange` matching specific strings against the keys."""
    strs = [strs] if isinstance(strs, str) else strs
    strs_to_exclude = (
        [strs_to_exclude]
        if isinstance(strs_to_exclude, str)
        else strs_to_exclude
    )

    def should_transform_pred(k: str | KeyPath, _):
      if isinstance(k, tuple) and isinstance(k[-1], tree_util.DictKey):
        leaf_key = k[-1].key
      elif isinstance(k, str):
        leaf_key = k
      else:
        raise ValueError(f"Unsupported key type: {type(k)}")

      matched = any([s in leaf_key for s in strs])
      to_exclude = any([s in leaf_key for s in strs_to_exclude])
      return matched and (not to_exclude)

    return cls(
        height=height,
        width=width,
        method=method,
        should_transform_pred=should_transform_pred,
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Normalize(kd.data.ElementWiseTransform):
  """Normalize an element."""

  in_vrange: tuple[float, float] = (0.0, 255.0)
  normalize_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
  normalize_std: tuple[float, float, float] = (1.0, 1.0, 1.0)

  dtype: Any = tf.float32

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    xnp = enp.lazy.get_xnp(element)
    dtype = enp.lazy.as_np_dtype(self.dtype)
    element = xnp.asarray(element, dtype=dtype)
    _, in_max = self.in_vrange
    element = element / in_max
    element = (element - self.normalize_mean) / self.normalize_std

    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RepeatFrames(kd.data.ElementWiseTransform):
  """Repeats frames so that they are divisible by `divisible_by`.

  For example, if the input is 11 frames and `divisible_by` is 5, then 15 sample
  points will be selected using the image resize operation's sampling grid and
  then rounded to the nearest integer.
  """

  divisible_by: int

  @typechecked
  def map_element(self, element: XArray["*b T H W C"]) -> XArray["*b T2 H W C"]:
    if enp.lazy.is_tf(element):
      # Tensorflow image resize only supports height and width dimensions so for
      # the time dimension we use gather.
      t = tf.shape(element)[-4]
      t2 = (
          tf.cast(tf.math.ceil(t / self.divisible_by), tf.int32)
          * self.divisible_by
      )
      indices = tf.image.resize(
          tf.reshape(tf.range(t), [1, -1, 1]), [1, t2], method="nearest"
      )[0, :, 0]
      return tf.gather(element, indices, axis=-4)
    elif enp.lazy.is_np(element) or enp.lazy.is_jax(element):
      xnp = enp.get_np_module(element)
      t = element.shape[-4]
      new_t = int(np.ceil(t / self.divisible_by) * self.divisible_by)
      indices = jax.image.resize(xnp.arange(t), (new_t,), method="nearest")
      return xnp.take(element, indices, axis=-4)
    else:
      raise ValueError(f"Unsupported type: {type(element)}")


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class FetchElementZero(kd.data.ElementWiseTransform):
  """Fetches an element: x[key][0, ...]."""

  @typechecked
  def map_element(self, element: XArray["B *stuff"]) -> XArray["*stuff"]:
    return element[0]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class SpacetimeToDepth(kd.data.ElementWiseTransform):
  """Fold the spatio-temporal dimension into depth."""

  space2depth_factor: int
  time2depth_factor: int

  @typechecked
  def map_element(
      self, element: XArray["*b T H W C"]
  ) -> XArray["*b T2 H2 W2 D"]:
    return einops.rearrange(
        element,
        "... (T t) (H h) (W w) D -> ... T H W (t h w D)",
        t=self.time2depth_factor,
        h=self.space2depth_factor,
        w=self.space2depth_factor,
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class VideoMAENormalization(kd.data.ElementWiseTransform):
  """Target normalization in VideoMAE."""

  patch_size: tuple[int, int, int] = (2, 16, 16)
  eps: float = 1e-6

  @typechecked
  def map_element(
      self, element: TfArray["*b T H W C"]
  ) -> TfArray["*b T H W C"]:
    t = element.shape[-4] // self.patch_size[0]
    h = element.shape[-3] // self.patch_size[1]
    w = element.shape[-2] // self.patch_size[2]
    element = einops.rearrange(
        element,
        "... (t p0) (h p1) (w p2) c -> ... (t h w) (p0 p1 p2) c",
        p0=self.patch_size[0],
        p1=self.patch_size[1],
        p2=self.patch_size[2],
    )
    mean = tf.math.reduce_mean(element, axis=-2, keepdims=True)
    std = tf.math.sqrt(tf.math.reduce_variance(element, axis=-2, keepdims=True))
    element = (element - mean) / (std + self.eps)

    element = einops.rearrange(
        element,
        "... (t h w) (p0 p1 p2) c -> ... (t p0) (h p1) (w p2) c",
        t=t,
        h=h,
        w=w,
        p0=self.patch_size[0],
        p1=self.patch_size[1],
        p2=self.patch_size[2],
    )
    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class AddStringField(grain.MapTransform):
  """Adds a string field to the batch."""

  key: str
  value: str
  overwrite_existing: bool = False

  @typechecked
  def map(self, batch: dict[str, Any]):
    """Adds a string field to the batch."""
    if not self.overwrite_existing and self.key in batch:
      raise ValueError(
          f"Key {self.key} already exists in the batch. Please set"
          " overwrite_existing to True if this is intended."
      )
    batch[self.key] = tf.constant(self.value, dtype=tf.string)
    return batch


def _expand_multi_index(
    multi_index: tf.Tensor,
    dim_to_indices: dict[int, tf.Tensor],
    total_dims: int,
) -> tf.Tensor:
  """Expands multi_index with values from dim_to_indices.

  Args:
    multi_index: A TensorFlow tensor representing the multi-index.
    dim_to_indices: A dictionary mapping dimensions to index values.
    total_dims: The total number of dimensions.

  Returns:
    A TensorFlow tensor with expanded multi-index values.
  """
  # Create a list to store the expanded indices
  expanded_indices = []

  # Iterate over dimensions using either the sampled indices in multi_index or
  # the duplicated indices in dim_to_indices
  duplicate_shape = tf.shape(next(iter(dim_to_indices.values())))[0]
  mi = 0
  for dim in range(total_dims):
    if dim in dim_to_indices:
      expanded_indices.append(
          tf.tile(dim_to_indices[dim], tf.shape(multi_index)[1:])
      )
    else:
      expanded_indices.append(tf.repeat(multi_index[mi], duplicate_shape))
      mi += 1
  # Stack the expanded indices along a new dimension
  expanded_indices = tf.stack(expanded_indices, axis=0)

  return expanded_indices


def _shuffle_and_partition(
    *,
    n_tokens: int,
    n_masked: int,
    shuffle_tokens: bool,
    seed: tf.Tensor,
) -> tf.Tensor:
  """Implements random shuffling and partitioning necessary for MAE for example.

  This is a non-batched tensorflow version of the function in
  scenic/projects/mfp/model_utils.py.

  Args:
    n_tokens: The number of tokens.
    n_masked: The number of tokens to mask. Must have 0 <= n_masked < n_tokens.
    shuffle_tokens: Whether to shuffle the tokens.
    seed: The random seed.

  Returns:
    Two arrays. The first one contains indices of masked tokens, and has
    shape [n_tokens, n_masked]. The second contains indices of unmasked tokens
    and has shape [n_tokens - n_masked].
  """
  if n_masked > n_tokens or n_masked < 0:
    raise ValueError(f"n_masked = {n_masked} should be >=0 and <{n_tokens}.")

  ids = tf.range(n_tokens, dtype=tf.int32)
  n_remainder = n_tokens - n_masked
  if shuffle_tokens and n_masked > 0:
    ids = tf.random.experimental.stateless_shuffle(ids, seed=seed)

  return tf.slice(ids, (0,), (n_remainder,))


def _unravel_index_inverse(multi_index: tf.Tensor, shape: tuple[int, ...]):
  """Calculates the flat index from a multidimensional index.

  Args:
    multi_index: A tuple of multidimensional indices.
    shape: The shape of the full multidimensional array.

  Returns:
    The index into a flattened version on the multidimensional array.
  """
  flat_index = tf.zeros(multi_index.shape[1:], multi_index.dtype)
  stride = 1
  for multi_index_index, dim_shape in zip(
      reversed(range(multi_index.shape[0])), reversed(shape)
  ):
    yi = multi_index[multi_index_index]
    flat_index += yi * stride
    stride *= dim_shape

  return flat_index


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandomDropTokens(grain.RandomMapTransform):
  """Drop tokens randomly. Typically preceded by time2depth and flattening.

  Supports both batched and non-batched inputs. Drops tokens from dimension -2.
  Assumes at most one batch dimension. Supports excluding indices from the
  drop, per batch element.
  """

  key: str
  key_exclude_indices: str | None = None
  subsampling_pattern: str = "random"
  drop_ratio: float
  shuffle_tokens: bool = True
  # If True, the indices will be offset for each element in the batch. For
  # example, if the batch has 2 elements with 4 tokens each, the first element
  # will have indices [0, 1, 2, 3] and the second element will have indices
  # [4, 5, 6, 7]. This is useful for cases where the batch contains clips of
  # the same video. If False, the indices will be relative to each batch element
  # e.g., they will be [0, 1, 2, 3] for both elements in the previous example.
  offset_indices: bool = False

  def remap_indices(
      self, feats, random_indices, n_tokens, feats_exclude_indices
  ):
    n_batch_dims = len(feats.shape) - 2

    if n_batch_dims == 0:
      feats = tf.expand_dims(feats, axis=0)
      random_indices = tf.expand_dims(random_indices, axis=0)
      feats_exclude_indices = tf.expand_dims(feats_exclude_indices, axis=0)

    n_shuffle_tokens = feats.shape[-2] - feats_exclude_indices.shape[-1]
    full_indices = tf.range(n_tokens, dtype=tf.int32)

    # Define a function to get available indices for a single batch element
    # This avoids complexities with batched boolean_mask output shapes
    def get_available_indices(exclude_idx_single):
      exclude_mask_single = tf.scatter_nd(
          indices=tf.expand_dims(exclude_idx_single, axis=-1),
          updates=tf.ones_like(exclude_idx_single, dtype=tf.bool),
          shape=[n_tokens],
      )
      available_mask_single = tf.logical_not(exclude_mask_single)
      return tf.boolean_mask(full_indices, available_mask_single)

    # Apply the function to each batch element
    available_indices_batched = tf.map_fn(
        get_available_indices,
        feats_exclude_indices,  # Iterate over the first dimension (batch)
        fn_output_signature=tf.TensorSpec(
            [n_shuffle_tokens], dtype=tf.int32
        ),  # Ensure consistent output shape
    )  # Shape [bs, n_shuffle_tokens]

    # Gather the final original indices using random_indices
    # (which index into available_indices_batched)
    random_indices = tf.gather(
        available_indices_batched, random_indices, batch_dims=1
    )  # Shape [bs, n_remainder]

    if n_batch_dims == 0:
      random_indices = tf.squeeze(random_indices, axis=0)

    return random_indices

  def random_map(self, features, rng):
    feats = features[self.key]

    if self.key_exclude_indices:
      feats_exclude_indices = features[self.key_exclude_indices]
    else:
      feats_exclude_indices = None

    # Drop tokens randomly.
    subsample_size = int(feats.shape[-2] * (1.0 - self.drop_ratio))

    n_batch_dims = len(feats.shape) - 2

    if n_batch_dims > 1:
      raise ValueError(
          f"Expected at most 1 batch dimension, got {n_batch_dims}."
      )

    if n_batch_dims == 0:
      bs = 1
    else:
      bs = feats.shape[0]

    def create_stateless_seed_from_rng(rng, batch_index):
      """Creates a unique stateless seed based on the rng and batch index."""

      if not isinstance(rng, tf.Tensor):
        rng = tf.constant(rng)

      new_seed_tensor = rng + tf.cast(batch_index, dtype=rng.dtype)

      return new_seed_tensor

    if feats_exclude_indices is not None:
      n_shuffle_tokens = feats.shape[-2] - feats_exclude_indices.shape[-1]
      if n_shuffle_tokens < subsample_size:
        raise ValueError(
            f"Expected feats_exclude_indices.shape[-1] <= {n_shuffle_tokens},"
            f" got {feats_exclude_indices.shape[-1]}."
        )
    else:
      n_shuffle_tokens = feats.shape[-2]

    n_masked_tokens = n_shuffle_tokens - subsample_size

    random_indices = tf.map_fn(
        lambda batch_index: _shuffle_and_partition(
            n_tokens=n_shuffle_tokens,
            n_masked=n_masked_tokens,
            shuffle_tokens=self.shuffle_tokens,
            seed=create_stateless_seed_from_rng(
                rng, batch_index
            ),  # Generate unique seed
        ),
        tf.range(bs),
    )

    if n_batch_dims == 0:
      random_indices = tf.reshape(random_indices, [-1])

    if feats_exclude_indices is not None:
      random_indices = self.remap_indices(
          feats, random_indices, feats.shape[-2], feats_exclude_indices
      )

    tokens = tf.gather(feats, random_indices, axis=-2, batch_dims=n_batch_dims)

    features[self.key] = tokens
    if n_batch_dims > 0 and self.offset_indices:
      random_indices += tf.range(bs)[..., None] * feats.shape[-2]
    features[f"{self.key}_indices"] = random_indices
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class SubsampleAndFlatten(grain.RandomMapTransform):
  """Subsample indices on some dimensions and flatten.

  For example if the input is [T, H, W, C] and sample_dims = [1, 2] and
  flatten_up_to = 3, then this will perform tube masking (i.e. selecting random
  spatial points and masking them across time) and then flatten to [N, C]. This
  also returns [N] indices of the unmasked tokens into the fully flattened array
  which would have shape [T*H*W].

  If sample_dims is changed to [0, 1, 2], this will do fully random masking.

  The default values will sample randomly on the first dimension and not
  flatten.

  Attributes:
    key: The key of the feature to subsample.
    drop_ratio: The ratio of tokens to drop.
    shuffle_tokens: Whether to shuffle the tokens before subsampling.
    sample_dims: The dimensions to subsample on.
    flatten_up_to: The dimension to flatten up to. All dimensions below this
      will be flattened into one.
  """

  key: str
  drop_ratio: float
  shuffle_tokens: bool = True
  sample_dims: tuple[int, ...] = (0,)
  flatten_up_to: int = 1

  def __post_init__(self):
    assert all(s < self.flatten_up_to for s in self.sample_dims)
    object.__setattr__(
        self,
        "sample_dim_to_dim",
        {si: i for i, si in enumerate(self.sample_dims)},
    )

  def random_map(
      self,
      element,
      rng: tf.Tensor,
  ):
    feats = element[self.key]
    # Drop tokens randomly from the sample dimensions.
    num_tokens = np.prod(tuple(feats.shape[si] for si in self.sample_dims))
    masked_size = int(num_tokens * self.drop_ratio)
    random_indices = _shuffle_and_partition(
        n_tokens=num_tokens,
        n_masked=masked_size,
        shuffle_tokens=self.shuffle_tokens,
        seed=rng,
    )
    # Duplicate for all values on other flattened dims.
    duplicate_dims = set(range(self.flatten_up_to)).difference(
        set(self.sample_dims)
    )
    # If there are some dimensions that need to be duplicated.
    if duplicate_dims:
      # Unravel the random indices on sample dims.
      random_indices = tf.unravel_index(
          indices=random_indices,
          dims=tuple(feats.shape[si] for si in self.sample_dims),
      )
      duplicate_dims_gen = (
          tf.range(feats.shape[si], dtype=tf.int32) for si in duplicate_dims
      )
      dup_indices = [
          tf.reshape(x, (-1,)) for x in tf.meshgrid(*duplicate_dims_gen)
      ]
      dim_to_indices = {i: d for i, d in zip(duplicate_dims, dup_indices)}
      expanded_multi_index = _expand_multi_index(
          random_indices, dim_to_indices, total_dims=self.flatten_up_to
      )
      # Ravel the indices.
      random_indices = _unravel_index_inverse(
          expanded_multi_index, feats.shape[: self.flatten_up_to]
      )
    # Flatten all dims.
    num_tokens = np.prod(
        tuple(feats.shape[si] for si in range(self.flatten_up_to))
    )
    feats = tf.reshape(
        feats,
        tf.concat(
            (tf.constant([num_tokens]), feats.shape[self.flatten_up_to :]),
            axis=0,
        ),
    )

    tokens = tf.gather(feats, random_indices, axis=0, batch_dims=0)

    element[self.key] = tokens
    element[f"{self.key}_indices"] = random_indices
    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class FlattenVideo(kd.data.ElementWiseTransform):
  """Fold middle (spatio-temporal) dimensions."""

  @typechecked
  def map_element(self, element: XArray["... T H W C"]) -> XArray["... N C"]:
    return einops.rearrange(element, "... t h w c -> ... (t h w) c")


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TimeChunkedFlattenVideo(kd.data.ElementWiseTransform):
  """Fold middle (spatio-temporal) dimensions with time chunking.

  Flattens middle dimensions except time, which can be partially folded into
  batch dimension.

  Example: [B, T, H, W, C] -> [B*T//time_chunk_size, time_chunk_size*H**W, C]

  (if time_chunk_size equals T, then this is equivalent to FlattenVideo)
  """

  time_chunk_size: int

  @typechecked
  def map_element(self, element: XArray["... T H W C"]) -> XArray["... N C"]:
    new_size = element.shape[-4] // self.time_chunk_size

    return einops.rearrange(
        element,
        "... (t time_chunk_size) h w c -> ... t (time_chunk_size h w) c",
        t=new_size,
        time_chunk_size=self.time_chunk_size,
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ReshapeSpatialDim(kd.data.ElementWiseTransform):
  h: int
  w: int

  @typechecked
  def map_element(self, element: TfArray["T d"]) -> TfArray["T h w"]:
    return tf.map_fn(lambda x: tf.reshape(x, [self.h, self.w]), element)


class MaskedVideoVisualization(grain.MapTransform):
  """Grain transform to visualize masked videos."""

  def __init__(
      self,
      sparse_patches_key: str,
      indices_key: str,
      output_key: str,
      grid_shape: tuple[int, int, int],  # T, H, W of the patch grid
      patch_shape: tuple[int, int, int],  # pt, ph, pw of each patch
      fill_value: float = 0.5,
      channels: int = 3,
  ):
    """Initializes the MaskedVideoVisualization transform.

    Args:
      sparse_patches_key: Key of the sparse patches tensor.
      indices_key: Key of the indices tensor.
      output_key: Key of the output visualization tensor.
      grid_shape: Shape of the patch grid (T, H, W).
      patch_shape: Shape of each patch (pt, ph, pw).
      fill_value: Value to fill the unmasked regions.
      channels: Number of channels in the output.
    """
    self.sparse_patches_key = sparse_patches_key
    self.indices_key = indices_key
    self.output_key = output_key
    self.grid_shape = grid_shape
    self.patch_shape = patch_shape
    self.fill_value = fill_value
    self.channels = channels

    # Precompute shapes and sizes
    self.t_grid, self.h_grid, self.w_grid = self.grid_shape
    self.pt, self.ph, self.pw = self.patch_shape
    self.n_total = self.t_grid * self.h_grid * self.w_grid
    self.p = self.pt * self.ph * self.pw * self.channels

    self.rearrange_patches_to_structured = lambda x: einops.rearrange(
        x,
        "... t h w (pt ph pw c) -> ... t h w pt ph pw c",
        pt=self.pt,
        ph=self.ph,
        pw=self.pw,
        c=self.channels,
    )
    self.rearrange_structured_to_final = lambda x: einops.rearrange(
        x, "... t h w pt ph pw c -> ... (t pt) (h ph) (w pw) c"
    )

  def map(self, features: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    """Reconstructs sparse patches for a single example."""
    sparse_patches = features[self.sparse_patches_key]
    indices = features[self.indices_key]

    p_actual = tf.shape(sparse_patches)[-1]

    tf.Assert(
        tf.equal(p_actual, self.p),
        [f"Patch dimension mismatch: expected {self.p}, got {p_actual}"],
    )

    # Create the full grid template filled with gray
    full_grid_template = tf.fill(
        (self.n_total, self.p),
        tf.cast(self.fill_value, sparse_patches.dtype),
    )

    # Prepare indices for tf.tensor_scatter_nd_update: shape [N_sparse, 1]
    # Each entry needs coordinate [grid_index]
    scatter_coords = tf.expand_dims(indices, axis=-1)  # Shape [N_sparse, 1]

    # Perform scatter update
    scattered_grid = tf.tensor_scatter_nd_update(
        full_grid_template, scatter_coords, sparse_patches
    )  # Updates template with patches

    # Reshape back to video format
    # (N_total, P) -> (T, H, W, P)
    video_patches = tf.reshape(
        scattered_grid,
        [self.t_grid, self.h_grid, self.w_grid, self.p],
    )
    # (T, H, W, P) -> (T, H, W, pt, ph, pw, C) -> (T*pt, H*ph, W*pw, C)
    video_structured = self.rearrange_patches_to_structured(video_patches)
    final_video = self.rearrange_structured_to_final(video_structured)

    features[self.output_key] = final_video
    return features
