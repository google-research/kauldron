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

from etils import enp
from etils.array_types import f32, ui8  # pylint: disable=g-multiple-import
from kauldron import kd
import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def graph_mode():
  # Run in graph mode since that is what the data pipeline will do
  with tf.Graph().as_default():
    yield


@enp.testing.parametrize_xnp(restrict=["np", "tnp"])
def test_value_range(xnp: enp.NpModule):
  vr = kd.data.ValueRange(
      key="values",
      in_vrange=(0.0, 255.0),
      vrange=(0.0, 1.0),
      clip_values=True,
  )
  before = {"values": xnp.array([-100, 0.0, 255.0, 400.0]), "other": 42}
  after = vr.map(before)
  assert after["other"] == before["other"]
  xnp.allclose(after["values"], xnp.array([0.0, 0.0, 1.0, 1.0]))


# Test which only check the array input/output specs
@pytest.mark.parametrize(
    "tr, in_spec, out_spec",
    (
        (
            kd.data.py.ValueRange(
                key="values",
                in_vrange=(0.0, 255.0),
                vrange=(0.0, 1.0),
                clip_values=True,
            ),
            ui8["2 3"],
            f32["2 3"],
        ),
        (
            kd.data.py.Rearrange(
                key="values",
                pattern="h w -> w h",
            ),
            ui8["2 3"],
            ui8["3 2"],
        ),
    ),
)
@enp.testing.parametrize_xnp(restrict=["np", "tnp"])
def test_transforms(
    xnp: enp.NpModule,
    tr,
    in_spec,
    out_spec,
):
  before = {
      "values": xnp.ones(_as_shape(in_spec.shape), in_spec.dtype.np_dtype)
  }
  after = tr.map(before)
  assert after["values"].shape == _as_shape(out_spec.shape)
  assert enp.lazy.as_np_dtype(after["values"].dtype) == out_spec.dtype.np_dtype


def _as_shape(shape: str) -> tuple[int, ...]:
  return tuple(int(d) for d in shape.split())


@enp.testing.parametrize_xnp(skip=["torch"])
def test_resize_with_size(xnp: enp.NpModule):
  vr = kd.data.py.Resize(
      key="img",
      size=(12, 12),
  )
  before = {"img": xnp.zeros((5, 5, 3), dtype=xnp.uint8)}
  after = vr.map(before)
  assert after["img"].shape == (12, 12, 3)


@enp.testing.parametrize_xnp(skip=["torch"])
def test_resize_with_min_size(xnp: enp.NpModule):
  vr = kd.data.py.Resize(
      key="img",
      min_size=10,
  )
  before = {"img": xnp.zeros((5, 6, 3), dtype=xnp.uint8)}
  after = vr.map(before)
  assert after["img"].shape == (10, 12, 3)


@enp.testing.parametrize_xnp(skip=["torch"])
def test_resize_with_max_size(xnp: enp.NpModule):
  vr = kd.data.py.Resize(
      key="img",
      max_size=5,
  )
  before = {"img": xnp.zeros((8, 10, 3), dtype=xnp.uint8)}
  after = vr.map(before)
  assert after["img"].shape == (4, 5, 3)


@enp.testing.parametrize_xnp(skip=["torch"])
def test_center_crop(xnp: enp.NpModule):
  vr = kd.data.py.CenterCrop(
      key="img",
      shape=(12, 12),
  )
  data = xnp.arange(16 * 16, dtype=xnp.uint8)
  data = xnp.reshape(data, (16, 16))
  expected_crop = data[2:14, 2:14]
  before = {"img": data}
  after = vr.map(before)
  assert after["img"].shape == (12, 12)
  xnp.allclose(after["img"], expected_crop)


@enp.testing.parametrize_xnp(skip=["torch"])
def test_center_crop_partial(xnp: enp.NpModule):
  vr = kd.data.py.CenterCrop(
      key="img",
      shape=(12, None),
  )
  data = xnp.arange(16 * 16, dtype=xnp.uint8)
  data = xnp.reshape(data, (16, 16))
  expected_crop = data[2:14, :]
  before = {"img": data}
  after = vr.map(before)
  assert after["img"].shape == (12, 16)
  xnp.allclose(after["img"], expected_crop)
