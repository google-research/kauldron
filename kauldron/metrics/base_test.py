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

import flax.struct
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
import numpy as np
import pytest


# --------- Test a custom metric -------
@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class IntAverage(base.Metric):
  x: kontext.Key = "batch.x"
  mask: kontext.Key = None

  @flax.struct.dataclass
  class State(base_state.AverageState):

    def compute(self):
      return int(super().compute())

  def get_state(self, x=None) -> IntAverage.State:
    return IntAverage.State.from_values(values=x)


def test_noop_metric():
  noop_metric = base.NoopMetric()
  assert not noop_metric.get_state().compute() and isinstance(
      noop_metric.get_state().compute(), dict
  )


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


def test_tree_reduce_state_and_parent():
  m = base.TreeReduce(metric=IntAverage())
  d = {
      "a": np.arange(12).reshape((3, 4)),
      "b": np.arange(8).reshape((8)),
  }
  state = m.get_state(x=d)
  empty = m.empty()
  assert isinstance(state, IntAverage.State)
  assert isinstance(empty, IntAverage.State)
  assert state.parent is m.metric
  assert empty.parent is m.metric


def test_tree_map_glob():
  m = base.TreeMap(metric=IntAverage(x="batch.**.d"))
  d = {
      "d": np.arange(12).reshape((3, 4)),
      "b": {
          "c": np.arange(8).reshape((8)),
          "d": np.arange(9).reshape((3, 1, 3)),
      },
  }
  y = m.get_state_from_context({"batch": d, "unused": d}).compute()
  assert y == {
      "d": 5,
      "b": {
          "d": 4,
      },
  }


def test_tree_reduce_glob():
  m = base.TreeReduce(metric=IntAverage(x="batch.**.d"))
  d = {
      "d": np.arange(2),
      "b": {
          "c": np.arange(20),
          "d": np.arange(4),
      },
  }
  y = m.get_state_from_context({"batch": d}).compute()
  assert y == 1  # Only d and b.d are included


def test_skip_if_missing():
  m = base.SkipIfMissing(metric=IntAverage(x="batch.d"))
  context = {
      "batch": {"d": np.arange(5)},
  }
  # should compute correct value if keys are present
  y = m.get_state_from_context(context).compute()
  assert y == 2

  # should not error and instead return empty state if keys are missing
  context = {"batch": {}}
  y = m.get_state_from_context(context).compute()
  assert y == 0


def test_error_if_state_is_not_flax_dataclass():
  # With decorator. Should not raise.
  class ValidMetric(base.Metric):  # pylint: disable=unused-variable

    @flax.struct.dataclass
    class State(base_state.State):
      a: int = 3

  with pytest.raises(TypeError, match="@flax.struct.dataclass"):

    class InvalidMetric(base.Metric):  # pylint: disable=unused-variable

      class State(base_state.State):
        a: int = 3
