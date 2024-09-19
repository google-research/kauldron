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

"""tf.data transforms."""

import dataclasses
from typing import Optional

from grain import tensorflow as grain
from kauldron.data.tf import transform_utils
from kauldron.data.transforms import base
from kauldron.typing import TfArray, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import tensorflow as tf


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseRandomTransform(
    base.ElementWiseRandomTransformBase, grain.RandomMapTransform
):
  """Base class for elementwise transforms."""

  # Wrap `random_map` to remove the `grain.META_FEATURES`
  random_map = transform_utils.wrap_map(
      base.ElementWiseRandomTransformBase.random_map
  )


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
    target_shape = transform_utils.get_target_shape(element, self.shape)
    # compute the range of the offset for the tf.slice
    offset_range = shape - target_shape
    clipped_offset_range = tf.clip_by_value(offset_range, 1, tf.int32.max)
    # randomly sample offsets from the desired range via modulo
    rand_int = tf.random.stateless_uniform(
        [shape.shape[0]], seed=seed, minval=None, maxval=None, dtype=tf.int32
    )
    offset = tf.where(offset_range > 0, rand_int % clipped_offset_range, 0)
    return tf.slice(element, offset, target_shape)  # crop
