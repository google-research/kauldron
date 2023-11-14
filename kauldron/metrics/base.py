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
import functools
from typing import Any, Mapping, TypeVar

import flax
import jax
from kauldron import kontext
from kauldron.metrics import base_state
from kauldron.typing import Float, PyTree  # pylint: disable=g-multiple-import,g-importing-member

_FnT = TypeVar("_FnT")


class Metric(abc.ABC):
  """Base class for metrics.

  Usage:

  ```python
  metric = kd.metrics.Norm()  # Initialize the metric
  state = metric.get_state(tensor=x)

  state = state.merge(other_state)  # States can be accumulated

  loss = state.compute()  # Get final value
  ```

  All metric implementations should be dataclasses that inherit from this class
  and:

  1) Overwrite the `Metric.State` class by inheriting from an appropriate
     `kd.metrics.State` that collects and aggregates the required information.
     In most cases this will either be:
      - `kd.metrics.AverageState` (for simple averaging of a value),
      - `kd.metrics.CollectingState` (for metrics that need to collect and
         concatenate model outputs over many batches)
  2) Define a set of `kd.kontext.Key` annotated fields that are used to set the
     paths for gathering information from the train/eval context.
  3) Override the `get_state(...)` method which should take arguments with the
     same names as the keys defined in 2). This method will usually be executed
     on device within a pmap. It should return an instance of `State` (1).
  4) Optionally override the `State.compute(...)` method which returns the
     final value of the metric. This method will be executed outside of
     jit/pmap and can thus make use of external libararies to perform its
     computation.
  """

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls.get_state = _link_metric_to_state(cls.get_state)
    cls.empty = _link_metric_to_state(cls.empty)
    if hasattr(cls, "compute"):
      # cls.compute is not Metric.compute:
      raise ValueError(
          "`Metric.compute` is deprecated. Instead, metrics should subclass"
          " `kd.metrics.State` and implement their own `.compute` function"
          f" accessing `self.parent`. Raised for {cls}"
      )

  @flax.struct.dataclass
  class State(base_state.State):
    pass

  @abc.abstractmethod
  def get_state(self, **kwargs) -> Metric.State:
    ...

  def empty(self) -> Metric.State:
    return self.State.empty()

  def _resolve_kwargs(self, context: Any) -> dict[kontext.Key, Any]:
    """Collects and returns the kwargs required for get_state from context."""
    return kontext.get_from_keys_obj(context, self, func=self.get_state)

  def get_state_from_context(self, context: Any) -> Metric.State:
    kwargs = self._resolve_kwargs(context)
    return self.get_state(**kwargs)

  def __call__(self, *, context: Any = None, **kwargs) -> PyTree[Float[""]]:
    if context is not None:
      if kwargs:
        raise TypeError(
            "Can either pass context or keyword arguments,"
            f"but got context and {kwargs.keys()}."
        )
      kwargs = self._resolve_kwargs(context)
    return self.get_state(**kwargs).compute()


def _link_metric_to_state(fn: _FnT) -> _FnT:
  """Set the `State.metric`."""
  if hasattr(fn, "_has_link_metric"):  # Function decorated already
    return fn

  @functools.wraps(fn)
  def new_get_state(self, *args, **kwargs):
    state = fn(self, *args, **kwargs)
    state = dataclasses.replace(state, parent=self)
    return state

  new_get_state._has_link_metric = True  # pylint: disable=protected-access
  return new_get_state


@flax.struct.dataclass
class TreeState(base_state.State):
  """Holds a pytree of metric states."""

  tree: Mapping["str", PyTree[base_state.State]] = flax.core.FrozenDict()

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
        lambda x, y: x.merge(y),
        self.tree,
        other.tree,
        is_leaf=base_state.State.isinstance,
    )
    return type(self)(tree=merged_tree)

  def compute(self) -> PyTree[Any]:  # pytype: disable=signature-mismatch  # jnp-array
    """Calls compute for all metric states in tree."""
    return jax.tree_map(
        lambda x: x.compute(), self.tree, is_leaf=base_state.State.isinstance
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

  def _resolve_kwargs(self, context: Any) -> dict[kontext.Key, Any]:
    # Use the key and get_state signature of self.metric instead of self
    return kontext.get_from_keys_obj(
        context, self.metric, func=self.metric.get_state
    )

  # Forwards `__kontext_keys__` so the keys can be extracted from the top-level
  # TODO(epot): Is it possible to remove `_resolve_kwargs` entirely by
  # having another `__kontext_func__` protocol to get the `func=` ?
  def __kontext_keys__(self) -> dict[kontext.Key, kontext.Key]:
    return kontext.get_keypaths(self.metric)


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

  return jax.tree_map(
      _fun_with_posargs, *posargs, is_leaf=base_state.State.isinstance
  )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TreeReduce(Metric):
  """Aggregates an inner metric over a pytree and returns a single result."""

  metric: Metric

  def get_state(self, **kwargs) -> base_state.State:
    state_tree = _tree_map_with_kwargs(self.metric.get_state, **kwargs)
    reduced_state = jax.tree_util.tree_reduce(
        lambda x, y: x.merge(y),
        state_tree,
        initializer=self.metric.empty(),
        is_leaf=base_state.State.isinstance,
    )
    return reduced_state

  def _resolve_kwargs(self, context: Any) -> dict[kontext.Key, Any]:
    # Use the key and get_state signature of self.metric instead of self
    return kontext.get_from_keys_obj(
        context, self.metric, func=self.metric.get_state
    )

  # Forwards `__kontext_keys__` so the keys can be extracted from the top-level
  def __kontext_keys__(self) -> dict[kontext.Key, kontext.Key]:
    return kontext.get_keypaths(self.metric)
