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

"""Test metrics logging."""

from __future__ import annotations

import dataclasses

import flax.struct
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.utils.kdash import dashboard_utils


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class IntAverage(base.Metric):
  x: kontext.Key = 'batch.x'
  mask: kontext.Key = None

  def __metric_names__(self):
    return ['average', 'twice_average']

  @flax.struct.dataclass
  class State(base_state.AverageState):

    def compute(self):
      return int(super().compute())

  def get_state(self, x=None) -> IntAverage.State:
    return dict(
        average=IntAverage.State.from_values(values=x),
        twice_average=IntAverage.State.from_values(values=2*x)
    )


def test_get_key():
  # Test when input = {}
  out = dashboard_utils._get_key({}, 'losses')
  assert not out

  # Test when input = {xx: None}
  out = dashboard_utils._get_key({'Average': None}, 'metrics')
  assert out == ['metrics/Average']

  out = dashboard_utils._get_key({'Average': None}, 'losses')
  assert out == ['losses/Average']

  # Test when input = {xx: Metrics}
  m = IntAverage()
  out = dashboard_utils._get_key({'Average': m}, 'metrics')
  expected_out = [f'metrics/Average/{name}' for name in m.__metric_names__()]
  assert out == expected_out
