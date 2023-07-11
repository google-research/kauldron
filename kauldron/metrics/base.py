# Copyright 2023 The kauldron Authors.
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

"""Base classes for defining metrics."""
from __future__ import annotations

import abc
from typing import Any

from clu import metrics as clu_metrics
from kauldron import core
from kauldron.typing import Float, PyTree  # pylint: disable=g-multiple-import,g-importing-member

State = clu_metrics.Metric


class Metric(abc.ABC):
  """Base class for metrics.

  All metric implementations should be dataclasses that inherit from this class
  and:

  1) Override the nested State class by inheriting from an appropriate
     `clu.metric.Metric` that collects and aggregates the required information.
     In most cases this will either be:
      - `clu.metrics.Average` (for simple averaging of a value),
      - `clu.metrics.CollectingMetric` (for metrics that need to collect and
         concatenate model outputs over many batches)
      - or an existing CLU metric to be wrapped.
  2) Define a set of `kd.typing.Key` annotated fields that are used to set the
     paths for gathering information from the train/eval context.
  3) Override the `get_state(...)` method which should take arguments with the
     same names as the keys defined in 2). This method will usually be executed
     on device within a pmap. It should return an instance of `State` (1).
  4) Optionally override the `compute(...)` method which takes an instance of
     `State` (e.g. produced by `get_state`) and returns the final value of
     the metric. This method will be executed outside of jit/pmap and can thus
     make use of external libararies to perform its computation.
  """

  class State(clu_metrics.Metric):
    pass

  @abc.abstractmethod
  def get_state(self, **kwargs) -> Metric.State:
    ...

  def compute(self, state: Metric.State) -> PyTree[Float[""]]:
    return state.compute()

  def empty(self) -> Metric.State:
    return self.State.empty()

  def get_state_from_context(self, context: Any) -> Metric.State:
    kwargs = core.resolve_kwargs(self, context, func=self.get_state)
    return self.get_state(**kwargs)

  def __call__(self, *, context: Any = None, **kwargs) -> PyTree[Float[""]]:
    if context is not None:
      if kwargs:
        raise TypeError(
            "Can either pass context or keyword arguments,"
            f"but got context and {kwargs.keys()}."
        )
      kwargs = core.resolve_kwargs(self, context, func=self.get_state)
    return self.compute(self.get_state(**kwargs))
