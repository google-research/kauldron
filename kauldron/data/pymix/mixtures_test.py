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

"""Test."""

from grain import python as grain
from kauldron import kd


def test_mix():
  ds1 = kd.data.py.DataSource(grain.RangeDataSource(0, 10, 1), shuffle=False)
  ds2 = kd.data.py.DataSource(grain.RangeDataSource(100, 110, 1), shuffle=False)

  ds = kd.data.py.Mix(
      [ds1, ds2],
      seed=0,
      num_epochs=1,
      num_workers=0,
      shuffle=False,
  )

  assert list(ds) == [
      0,
      100,
      1,
      101,
      2,
      102,
      3,
      103,
      4,
      104,
      5,
      105,
      6,
      106,
      7,
      107,
      8,
      108,
      9,
      109,
  ]

  # By default, deterministic shuffling
  ds = kd.data.py.Mix(
      [ds1, ds2],
      seed=0,
      num_epochs=1,
      num_workers=0,
  )
  assert list(ds) == [
      2,
      101,
      4,
      109,
      5,
      106,
      104,
      103,
      107,
      1,
      102,
      105,
      0,
      8,
      6,
      9,
      108,
      100,
      7,
      3,
  ]
