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

"""Test."""

import copy
from typing import Callable
from unittest import mock

from grain import tensorflow as grain
from kauldron import kd
from kauldron.contrib.data import preprocessing
import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def graph_mode():
  # Run in graph mode since that is what the data pipeline will do
  with tf.Graph().as_default():
    yield


def test_add_constants():
  values = {"new": 2}
  el = kd.contrib.data.AddConstants(values=values)
  before = {"old": 1}
  after = el.map(before)
  assert set(after.keys()) == {"old", "new"}
  assert after["old"] == before["old"]
  assert after["new"] == values["new"]


def test_add_constants_overwrite_raises():
  el = kd.contrib.data.AddConstants(values={"oops": 3})
  before = {"old": 1, "oops": 2}
  with pytest.raises(KeyError):
    el.map(before)


def test_batch_random_drop_tokens(
    drop_ratio: float = 1 / 4,
    shape: tuple[int, int, int] = (3, 32, 21),
    target_shape: tuple[int, int, int] = (3, (32 - 32 // 4), 21),
):
  cc = kd.contrib.data.RandomDropTokens(key="values", drop_ratio=drop_ratio)
  before = {
      "values": tf.reshape(tf.range(np.prod(shape)), shape),
      grain.INDEX: tf.constant(0),
  }
  after = cc.random_map(copy.copy(before), rng=(0, 0))
  assert before["values"].shape == shape
  assert after["values"].shape == target_shape

  after = cc.random_map(copy.copy(before), rng=(1, 0))
  assert before["values"].shape == shape
  assert after["values"].shape == target_shape


def test_random_flip_left_right_video():
  cc = kd.contrib.data.RandomFlipLeftRightVideo(
      key="values",
  )
  before = {
      "values": tf.reshape(tf.range(150), (2, 5, 5, 3)),
      grain.INDEX: tf.constant(0),
  }
  after = cc.random_map(before, seed=(0, 0))  # does flip
  assert after["values"].shape == (2, 5, 5, 3)

  after = cc.random_map(before, seed=(1, 0))  # does not flip
  assert after["values"].shape == (2, 5, 5, 3)


def get_mocked_shuffle(
    fake_shuffled: tf.Tensor,
) -> Callable[..., tf.Tensor]:
  def mocked_shuffle(*args, **kwargs) -> tf.Tensor:
    del args, kwargs
    return fake_shuffled

  return mocked_shuffle


def test_subsample_and_flatten():
  inputs = {
      "video": tf.reshape(tf.range(27), (3, 3, 3, 1)),  # [T, H, W, C]
  }

  # Tube sampling
  transform = kd.contrib.data.SubsampleAndFlatten(
      key="video",
      drop_ratio=0.4,
      sample_dims=[1, 2],
      flatten_up_to=3,
  )

  # Check the transform returns the correct shaped outputs.
  outputs = transform.random_map(inputs.copy(), rng=(0, 0))
  expected_length = 18  # (9 - int(0.4 * 9)) * 3
  assert outputs["video"].shape == (expected_length, 1)
  assert outputs["video_indices"].shape == (expected_length,)

  # Mock the shuffle and check that everything else is as expected.
  with mock.patch.object(
      preprocessing, "_shuffle_and_partition", autospec=True
  ) as mocked:
    mocked.side_effect = get_mocked_shuffle(tf.constant([6, 3]))
    outputs = transform.random_map(inputs.copy(), rng=(0, 0))
    with tf.compat.v1.Session() as sess:
      outputs = sess.run(outputs)
  # Values should be for all values of t i.e. 0, 1, 2. This means we get the
  # spatial indices + t * H * W, i.e. +0, +9 and +18.
  assert all(outputs["video_indices"] == np.array([6, 15, 24, 3, 12, 21]))
  # Because we set the inputs using range the indices are the same as the
  # values (but without the channel dimension).
  assert all(outputs["video"] == np.array([[6], [15], [24], [3], [12], [21]]))

  # Random sampling
  transform = kd.contrib.data.SubsampleAndFlatten(
      key="video",
      drop_ratio=0.4,
      sample_dims=[0, 1, 2],
      flatten_up_to=3,
  )

  # Check the transform returns the correct shaped outputs.
  outputs = transform.random_map(inputs.copy(), rng=(0, 0))
  expected_length = 17  # 27 - int(0.4 * 27)
  assert outputs["video"].shape == (expected_length, 1)
  assert outputs["video_indices"].shape == (expected_length,)

  # Mock the shuffle and check that everything else is as expected.
  with mock.patch.object(
      preprocessing, "_shuffle_and_partition", autospec=True
  ) as mocked:
    mocked.side_effect = get_mocked_shuffle(tf.constant([6, 3]))
    outputs = transform.random_map(inputs.copy(), rng=(0, 0))
    with tf.compat.v1.Session() as sess:
      outputs = sess.run(outputs)
  # Because we set the inputs using range the indices are the same as the
  # values (but without the channel dimension).
  assert all(outputs["video_indices"] == np.array([6, 3]))
  assert all(outputs["video"] == np.array([[6], [3]]))


def test_random_resize():
  batch_size, height, width, channels = 5, 4, 6, 3
  # Output shape should be within the range of the scale factors.
  min_scale_factor = 0.5
  max_scale_factor = 1.5
  num_elements = batch_size * height * width * channels
  inputs = {
      "image": tf.reshape(
          tf.range(num_elements), (batch_size, height, width, channels)
      ),
      grain.INDEX: tf.constant(0),  # Placeholder for pygrain index
  }
  # Verify that resizing is applied when resizing probability is 1.
  prob = 1.0

  transform = kd.contrib.data.RandomResize(
      key="image",
      min_scale_factor=min_scale_factor,
      max_scale_factor=max_scale_factor,
      prob=prob,
  )
  outputs = transform.random_map(inputs.copy(), seed=(0, 0))
  with tf.compat.v1.Session() as sess:
    outputs = sess.run(outputs)
  new_height, new_width = outputs["image"].shape[1:3]

  # Compute min and max height and width based on the scale factors.
  min_height = int(height * min_scale_factor)
  max_height = int(height * max_scale_factor)
  min_width = int(width * min_scale_factor)
  max_width = int(width * max_scale_factor)
  assert min_height <= new_height <= max_height
  assert min_width <= new_width <= max_width

  # Setting prob to 0.0 should return the same shape as the input as resizing
  # is never applied.
  prob = 0.0
  transform = kd.contrib.data.RandomResize(
      key="image",
      min_scale_factor=min_scale_factor,
      max_scale_factor=max_scale_factor,
      prob=prob,
  )
  outputs = transform.random_map(inputs.copy(), seed=(0, 0))
  with tf.compat.v1.Session() as sess:
    outputs = sess.run(outputs)
  # Check that the output shape is the same as the input shape when prob = 0.0
  new_height, new_width = outputs["image"].shape[1:3]
  assert new_height == height
  assert new_width == width


@pytest.mark.parametrize(
    "size, divisible_by, expected_size, expected_out, use_tf",
    [
        (3, 2, 4, [0.0, 1.0, 1.0, 2.0], True),
        (3, 2, 4, [0.0, 1.0, 1.0, 2.0], False),
        (11, 5, 15, [0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0,
                     5.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0], True),
        (11, 5, 15, [0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0,
                     5.0, 6.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0], False),
    ],
)
def test_repeat_frames(size, divisible_by, expected_size, expected_out, use_tf):
  op = kd.contrib.data.RepeatFrames(key="video", divisible_by=divisible_by)
  if use_tf:
    video = tf.cast(tf.reshape(tf.range(size), [size, 1, 1, 1]), tf.float32)
  else:
    video = np.arange(size, dtype=np.float32).reshape((size, 1, 1, 1))
  ex = {"video": video}
  out = op.map(ex)
  if use_tf:
    with tf.compat.v1.Session() as sess:
      out = sess.run(out)
  out = out["video"]
  assert out.shape == (expected_size, 1, 1, 1)
  np.testing.assert_array_equal(out[:, 0, 0, 0], expected_out)
