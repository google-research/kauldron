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
import dataclasses
from typing import Any, Mapping

from clu import metrics as clu_metrics
import flax
import jax
from kauldron.typing import Float, Key, PyTree  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import core

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

  @flax.struct.dataclass
  class State(clu_metrics.Metric):
    pass

  @abc.abstractmethod
  def get_state(self, **kwargs) -> Metric.State:
    ...

  def compute(self, state: Metric.State) -> PyTree[Float[""]]:
    return state.compute()

  def empty(self) -> Metric.State:
    return self.State.empty()

  def resolve_kwargs(self, context: Any) -> dict[Key, Any]:
    """Collects and returns the kwargs required for get_state from context."""
    return core.resolve_kwargs(self, context, func=self.get_state)

  def get_state_from_context(self, context: Any) -> Metric.State:
    kwargs = self.resolve_kwargs(context)
    return self.get_state(**kwargs)

  def __call__(self, *, context: Any = None, **kwargs) -> PyTree[Float[""]]:
    if context is not None:
      if kwargs:
        raise TypeError(
            "Can either pass context or keyword arguments,"
            f"but got context and {kwargs.keys()}."
        )
      kwargs = self.resolve_kwargs(context)
    return self.compute(self.get_state(**kwargs))


@flax.struct.dataclass
class TreeState(clu_metrics.Metric):
  """Holds a pytree of metric states."""

  tree: Mapping["str", PyTree[clu_metrics.Metric]] = flax.core.FrozenDict()

  @classmethod
  def empty(cls) -> TreeState:
    return cls()

  def merge(self, other: TreeState) -> TreeState:
    """Merge two trees of metric states."""
    if not isinstance(other, type(self)):
      raise TypeError(f"Cannot merge {type(self)} with {type(other)}.")

    # handle the case of empty trees
    if not self.tree:
      return other
    if not other.tree:
      return self

    merged_tree = jax.tree_map(
        lambda x, y: x.merge(y), self.tree, other.tree, is_leaf=_is_metric_state
    )
    return type(self)(tree=merged_tree)

  def reduce(self) -> TreeState:
    """Reduces all metric states in the tree (merge along first dimension)."""
    reduced_tree = jax.tree_map(
        lambda x: x.reduce(), self.tree, is_leaf=_is_metric_state
    )
    return type(self)(tree=reduced_tree)

  def compute(self) -> PyTree[Any]:  # pytype: disable=signature-mismatch  # jnp-array
    """Calls compute for all metric states in tree."""
    return jax.tree_map(
        lambda x: x.compute(), self.tree, is_leaf=_is_metric_state
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TreeMap(Metric):
  """Maps an inner metric to a pytree and returns a pytree of results.

  Example:
    `train_metrics["param_norm"] = TreeMap(metric=Norm(tensor="params"))` would
    separately track and return the norm of all parameters of the model.


  Attributes:
    metric: Any metric to apply to the leaves of the tree. Also uses the keys of
      that metric to resolve the kwargs.
  """

  metric: Metric

  @flax.struct.dataclass
  class State(TreeState):
    pass

  def get_state(self, **kwargs):
    state_tree = _tree_map_with_kwargs(self.metric.get_state, **kwargs)
    return self.State(state_tree)

  def resolve_kwargs(self, context: Any) -> dict[Key, Any]:
    # Use the key and get_state signature of self.metric instead of self
    return core.resolve_kwargs(self.metric, context, func=self.metric.get_state)

  def compute(self, state: TreeMap.State):
    return _tree_map_with_kwargs(self.metric.compute, state=state.tree)


def _tree_map_with_kwargs(fun, **kwargs):
  """Same as jax.tree_map but taking and passing trees to fun as kwargs."""
  # This function will be passed the keyword arguments meant for the inner
  # metric. Except that the these args are now pytrees instead of tensors.
  # So all that is left to do is to tree-map the metric.get_state function
  # over a zip of these trees.
  # jax.tree_map supports this but only as pos-args and not with kw-args.
  # So we serialize the kwargs into a tuple of positional args, call tree_map
  # and for each function call turn them back into kwargs
  argnames = [k for k, v in kwargs.items() if v is not None]
  posargs = [v for _, v in kwargs.items() if v is not None]

  def _fun_with_posargs(*args):
    kwargs = dict(zip(argnames, args))
    return fun(**kwargs)

  return jax.tree_map(_fun_with_posargs, *posargs, is_leaf=_is_metric_state)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TreeReduce(Metric):
  """Aggregates an inner metric over a pytree and returns a single result."""

  metric: Metric

  def get_state(self, **kwargs) -> clu_metrics.Metric:
    state_tree = _tree_map_with_kwargs(self.metric.get_state, **kwargs)
    reduced_state = jax.tree_util.tree_reduce(
        lambda x, y: x.merge(y),
        state_tree,
        initializer=self.metric.empty(),
        is_leaf=_is_metric_state,
    )
    return reduced_state

  def resolve_kwargs(self, context: Any) -> dict[Key, Any]:
    # Use the key and get_state signature of self.metric instead of self
    return core.resolve_kwargs(self.metric, context, func=self.metric.get_state)

  def compute(self, state: clu_metrics.Metric):
    return self.metric.compute(state)


def _is_metric_state(x: Any) -> bool:
  """Check if x is a valid Metric.State. Used as is_leaf fun in tree_map."""
  return isinstance(x, clu_metrics.Metric)
