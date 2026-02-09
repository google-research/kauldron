# Copyright 2026 The kauldron Authors.
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

"""Preprocessing ops that use scenic video ops."""

from __future__ import annotations

import dataclasses

from kauldron import kd
from kauldron.typing import TfArray, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import tensorflow as tf


def deterministic_crop(images, size, spatial_idx):
  """Takes a deterministic crop of input images.

  Args:
    images: `Tensor` of shape shape [t, h, w, c]
    size: Integer ; size of height and width to crop the images.
    spatial_idx: 0, 1, or 2 for left, center, and right crop if width is larger
      than height. Or 0, 1, or 2 for top, center, and bottom crop if height is
      larger than width.

  Returns:
    cropped: `Tensor` of shape [t, crop_size, crop_size, c]
  """
  assert spatial_idx in [0, 1, 2]
  height, width = tf.shape(images)[1], tf.shape(images)[2]

  y_offset = tf.cast(tf.math.ceil((height - size) / 2), tf.int32)
  x_offset = tf.cast(tf.math.ceil((width - size) / 2), tf.int32)

  if height > width:
    if spatial_idx == 0:
      y_offset = 0
    elif spatial_idx == 2:
      y_offset = height - size
  else:
    if spatial_idx == 0:
      x_offset = 0
    elif spatial_idx == 2:
      x_offset = width - size

  cropped = tf.slice(images, [0, y_offset, x_offset, 0], [-1, size, size, -1])

  return cropped


def three_spatial_crops(images, crop_size):
  """Returns three spatial crops of the same frame, as done by SlowFast.

  This enables testing using the same protocol as prior works. ie
  (https://arxiv.org/abs/1812.03982, https://arxiv.org/abs/1904.02811,
   https://arxiv.org/abs/2004.04730)
  If width > height, takes left, centre and right crop.
  If height > width, takes top, middle and bottom crop.

  Args:
    images: `Tensor` of shape [t, h, w, c]
    crop_size: The size to crop from the images

  Returns:
    `Tensor` of shape [3 * t, h, w, c]
  """

  result = []
  for spatial_index in range(3):
    images_cropped = deterministic_crop(images, crop_size, spatial_index)
    result.append(images_cropped)

  return tf.concat(result, axis=0)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ThreeSpatialCrop(kd.data.ElementWiseTransform):
  """Returns three spatial crops of the same frame, as done by SlowFast.

  This enables testing using the same protocol as prior works. ie
  (https://arxiv.org/abs/1812.03982, https://arxiv.org/abs/1904.02811,
   https://arxiv.org/abs/2004.04730)
  If width > height, takes left, centre and right crop.
  If height > width, takes top, middle and bottom crop.

  Attributes:
    shape: An integer describing the target shape of the crop. Entries can be
      also be None to keep the original shape of the data in that dim.
  """

  shape: int

  @typechecked
  def map_element(self, element: TfArray["..."]) -> TfArray["..."]:
    return three_spatial_crops(element, self.shape)
