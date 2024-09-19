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
from kauldron import kd


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
