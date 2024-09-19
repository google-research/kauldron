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

from etils import enp
from grain import tensorflow as grain
from kauldron import kd
import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def graph_mode():
  # Run in graph mode since that is what the data pipeline will do
  with tf.Graph().as_default():
    yield


@enp.testing.parametrize_xnp(restrict=["np", "tnp"])
def test_value_range(xnp: enp.NpModule):
  vr = kd.data.tf.ValueRange(
      key="values",
      in_vrange=(0.0, 255.0),
      vrange=(0.0, 1.0),
      clip_values=True,
  )
  before = {"values": xnp.array([-100, 0.0, 255.0, 400.0]), "other": 42}
  after = vr.map(before)
  assert after["other"] == before["other"]
  xnp.allclose(after["values"], xnp.array([0.0, 0.0, 1.0, 1.0]))


def test_center_crop():
  cc = kd.data.tf.CenterCrop(
      key="values",
      shape=(2, 2),
  )
  before = {"values": tf.constant(np.arange(12).reshape(3, 4))}
  after = cc.map(before)
  assert after["values"].shape == (2, 2)
  tf.debugging.assert_equal(after["values"], before["values"][0:2, 1:3])


def test_center_crop_dynamic_dim():
  cc = kd.data.tf.CenterCrop(
      key="values",
      shape=(3, None, 1),
  )
  before = {"values": tf.zeros((5, 5, 5))}
  after = cc.map(before)
  assert after["values"].shape == (3, 5, 1)


def test_random_crop():
  cc = kd.data.tf.RandomCrop(
      key="values",
      shape=(7, None, 3),
  )
  before = {
      "values": tf.zeros((16, 5, 3)),
      grain.INDEX: tf.constant(0),
  }
  after = cc.random_map(before, seed=(0, 0))
  assert after["values"].shape == (7, 5, 3)


def test_resize():
  # 0 0 1 1
  # 0 0 1 1
  # 1 0 1 0
  # 0 1 0 1
  img = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
  img = img.reshape((1, 1, 1, 4, 4, 1))
  img_int = tf.convert_to_tensor(img, dtype=tf.int32)
  img_float = tf.convert_to_tensor(img, dtype=tf.float32)

  r = kd.data.tf.Resize(key="image", height=2, width=2, method="AUTO")

  # for int images preserve dtype and use nearest
  before = {"image": img_int}
  after = r.map(before)
  assert after["image"].shape == (1, 1, 1, 2, 2, 1)
  assert after["image"].dtype == tf.int32
  tf.debugging.assert_equal(
      after["image"][0, 0, 0, :, :, 0], tf.constant([[0, 1], [1, 1]])
  )

  # for float images average
  before = {"image": img_float}
  after = r.map(before)
  assert after["image"].shape == (1, 1, 1, 2, 2, 1)
  assert after["image"].dtype == tf.float32
  tf.debugging.assert_near(
      after["image"][0, 0, 0, :, :, 0], tf.constant([[0.0, 1.0], [0.5, 0.5]])
  )


def test_resize_small():
  img = np.array([
      [0, 0, 1, 1],
      [0, 0, 1, 1],
      [1, 0, 1, 0],
      [0, 1, 0, 1],
      [2, 2, 3, 3],
      [2, 2, 3, 3],
  ])
  img = img.reshape((1, 1, 1, 6, 4, 1))
  img_int = tf.convert_to_tensor(img, dtype=tf.int32)
  img_float = tf.convert_to_tensor(img, dtype=tf.float32)

  r = kd.data.tf.ResizeSmall(key="image", smaller_size=2, method="AUTO")

  # for int images preserve dtype and use nearest
  before = {"image": img_int}
  after = r.map(before)
  assert after["image"].shape == (1, 1, 1, 3, 2, 1)
  assert after["image"].dtype == tf.int32
  tf.debugging.assert_equal(
      after["image"][0, 0, 0, :, :, 0], tf.constant([[0, 1], [1, 1], [2, 3]])
  )

  # for float images average
  before = {"image": img_float}
  after = r.map(before)
  assert after["image"].shape == (1, 1, 1, 3, 2, 1)
  assert after["image"].dtype == tf.float32
  tf.debugging.assert_near(
      after["image"][0, 0, 0, :, :, 0],
      tf.constant([[0.0, 1.0], [0.5, 0.5], [2.0, 3.0]]),
  )
