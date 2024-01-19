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

"""Tests for metrics base classes."""
from __future__ import annotations

import dataclasses

from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
import numpy as np


# --------- Test a custom metric -------
@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class IntAverage(base.Metric):
  x: kontext.Key = "batch.x"

  class State(base_state.AverageState):

    def compute(self):
      return int(super().compute())

  def get_state(self, x=None) -> IntAverage.State:
    return IntAverage.State.from_values(values=x)


def test_avgerage_x_metric():
  m = IntAverage()
  x = np.arange(12).reshape((3, 4))
  # interface 1
  s1 = m.get_state(x=x)
  y1 = s1.compute()
  assert y1 == 5

  # with merging
  e = m.empty()
  y1 = e.merge(s1).compute()
  assert y1 == 5

  # interface 2
  y2 = m.get_state_from_context({"batch": {"x": x}}).compute()
  assert y2 == 5

  # interface 3
  y3 = m(x=x)
  assert y3 == 5


def test_tree_map():
  m = base.TreeMap(metric=IntAverage())
  d = {
      "a": np.arange(12).reshape((3, 4)),
      "b": {
          "c": np.arange(8).reshape((8)),
          "d": np.arange(9).reshape((3, 1, 3)),
      },
  }
  y = m(x=d)
  assert y == {"a": 5, "b": {"c": 3, "d": 4}}


def test_tree_reduce():
  m = base.TreeReduce(metric=IntAverage())
  d = {
      "a": np.arange(12).reshape((3, 4)),
      "b": {
          "c": np.arange(8).reshape((8)),
          "d": np.arange(9).reshape((3, 1, 3)),
      },
  }
  y = m(x=d)
  assert y == 4  # IntAverage of a, c, and d
