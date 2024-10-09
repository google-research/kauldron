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
     jit/pmap and can thus make use of external libraries to perform its
     computation.
  """

  # Hack to allow config to set `loss.__qualname__ = 'MyLoss'`
  __qualname__: str

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
    # nested State classes are defined with @flax.struct.dataclass decorator
    if not _is_flax_dataclass(cls.State):
      raise TypeError(
          f"Metric state {cls.__name__}.State must be defined with "
          "@flax.struct.dataclass decorator."
      )

  def __metric_names__(self) -> list[str] | None:
    """Returns the metric names.

    Returns:
      None: If the metric is a scalar.
      list[str]: A list of sub-metric names if the metric is a dictionary.
    """
    return None

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
    return kontext.resolve_from_keyed_obj(context, self, func=self.get_state)

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
      state = self.get_state_from_context(context)
    else:
      state = self.get_state(**kwargs)
    return state.compute()


def _link_metric_to_state(fn: _FnT) -> _FnT:
  """Set the `State.metric`."""
  if hasattr(fn, "_has_link_metric"):  # Function decorated already
    return fn

  @functools.wraps(fn)
  def new_get_state(self, *args, **kwargs):
    state = fn(self, *args, **kwargs)
    if state.parent is base_state.EMPTY:  # pylint: disable=protected-access
      # preserve the parent if it is already set
      # this is important for wrapper metrics like `kd.metrics.TreeReduce`
      # where the parent should remain set to the inner metric.
      state = dataclasses.replace(state, parent=self)
    return state

  new_get_state._has_link_metric = True  # pylint: disable=protected-access
  return new_get_state


def _is_flax_dataclass(cls) -> bool:
  # Note: We are checking __dict__ instead of using getattr to exclude the
  # `_flax_dataclass` attribute of the parent class base_state.State.
  return dataclasses.is_dataclass(cls) and cls.__dict__.get(
      "_flax_dataclass", False
  )


class NoopMetric(Metric):
  """Metric that does nothing. Can be used in sweeps to remove a metric."""

  @flax.struct.dataclass
  class State(base_state.EmptyState):
    pass

  def get_state(self, **kwargs: Any) -> NoopMetric.State:
    del kwargs
    return self.State.empty()


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

    merged_tree = jax.tree.map(
        lambda x, y: x.merge(y),
        self.tree,
        other.tree,
        is_leaf=base_state.State.isinstance,
    )
    return type(self)(tree=merged_tree)

  def compute(self) -> PyTree[Any]:  # pytype: disable=signature-mismatch  # jnp-array
    """Calls compute for all metric states in tree."""
    return jax.tree.map(
        lambda x: x.compute(), self.tree, is_leaf=base_state.State.isinstance
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class _TreeMetric(Metric):
  """Base class for metrics that are applied to a pytree."""

  metric: Metric

  # TODO(epot): Is it possible to remove `_resolve_kwargs` entirely by
  # having another `__kontext_func__` protocol to get the `func=` ?
  def _resolve_kwargs(self, context: Any) -> dict[kontext.Key, Any]:
    # Use the key and get_state signature of self.metric instead of self
    return kontext.resolve_from_keyed_obj(
        context, self, func=self.metric.get_state
    )

  def _get_tree_state(self, **kwargs) -> PyTree[jax.Array]:
    """Extract the tree of metric states."""
    # Filter `None` keys as they are not passed as `kwargs`.
    # Match filtering in `kontext.resolve_from_keyed_obj`
    keypaths = {k: v for k, v in self._glob_keypaths.items() if v is not None}

    kwargs = jax.tree.map(
        lambda path, ctx: kontext.filter_by_path(
            ctx,
            # Truncate the path to exclude the already-selected part:
            # `params.**.bias` -> `**.bias`
            # The non-glob prefix (`params`) is taken care of already
            # because it is returned by the __kontext_keys__ method.
            path.relative_to(path.first_non_glob_parent),
        ),
        keypaths,
        kwargs,
    )

    # If no values were found matching the metric keys, raise an explicit
    # KeyError for better understanding and to make this compatible with
    # `SkipIfMissing`.
    non_empty_values = [v for v in kwargs.values() if v is not None]
    if not non_empty_values:
      raise KeyError(
          f"{self.__class__.__name__}: No keys found matching any of the given"
          f" {keypaths=}"
      )

    return _tree_map_with_kwargs(self.metric.get_state, **kwargs)

  # Forwards `__kontext_keys__` so the keys can be extracted from the top-level
  def __kontext_keys__(self) -> dict[kontext.Key, kontext.Key]:
    return jax.tree.map(
        lambda glob_path: glob_path.first_non_glob_parent,
        self._glob_keypaths,
    )

  @property
  def _glob_keypaths(self):
    return jax.tree.map(
        kontext.GlobPath.from_str,
        kontext.get_keypaths(self.metric),
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TreeMap(_TreeMetric):
  """Maps an inner metric to a pytree and returns a pytree of results.

  Example:
    `train_metrics["param_norm"] = TreeMap(metric=Norm(tensor="params"))` would
    separately track and return the norm of all parameters of the model.


  Attributes:
    metric: Any metric to apply to the leaves of the tree. Also uses the keys of
      that metric to resolve the kwargs.
  """

  @flax.struct.dataclass
  class State(TreeState):
    pass

  def get_state(self, **kwargs):
    state_tree = self._get_tree_state(**kwargs)
    return self.State(state_tree)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TreeReduce(_TreeMetric):
  """Applies a metric to a pytree and returns the aggregated result.

  The given metric defines the aggregation method.
  """

  def get_state(self, **kwargs) -> base_state.State:
    state_tree = self._get_tree_state(**kwargs)
    reduced_state = jax.tree.reduce(
        lambda x, y: x.merge(y),
        state_tree,
        initializer=self.metric.empty(),
        is_leaf=base_state.State.isinstance,
    )
    return reduced_state

  def empty(self) -> base_state.State:
    return self.metric.empty()


def _tree_map_with_kwargs(fun, **kwargs):
  """Same as jax.tree.map but taking and passing trees to fun as kwargs."""
  # This function will be passed the keyword arguments meant for the inner
  # metric. Except that the these args are now pytrees instead of tensors.
  # So all that is left to do is to tree-map the metric.get_state function
  # over a zip of these trees.
  # jax.tree.map supports this but only as pos-args and not with kw-args.
  # So we serialize the kwargs into a tuple of positional args, call tree_map
  # and for each function call turn them back into kwargs
  argnames = [k for k, v in kwargs.items() if v is not None]
  posargs = [v for _, v in kwargs.items() if v is not None]

  def _fun_with_posargs(*args):
    kwargs = dict(zip(argnames, args))
    return fun(**kwargs)

  return jax.tree.map(
      _fun_with_posargs, *posargs, is_leaf=base_state.State.isinstance
  )


@dataclasses.dataclass(frozen=True, eq=True)
class SkipIfMissing(Metric):
  """Skip this metric if any of the keys are missing.

  This can be useful for example for metrics that are only defined for a
  subset of the datasets, or for metrics of gradients that would fail during
  evaluation.

  Usage:
   cfg.train_metrics["optional_metric"] = kd.metrics.SkipIfMissing(
       kd.metrics.Norm(tensor="grads.encoder")
    )

  Attributes:
    metric: The metric to skip if any of its kontext-keys are missing.
  """

  metric: Metric

  def _resolve_kwargs(self, context: Any) -> dict[kontext.Key, Any]:
    # Use the key and get_state signature of self.metric instead of self
    return kontext.resolve_from_keyed_obj(
        context, self, func=self.metric.get_state
    )

  def get_state(self, **kwargs) -> Metric.State:
    return self.metric.get_state(**kwargs)

  def get_state_from_context(self, context: Any) -> Metric.State:
    try:
      kwargs = self._resolve_kwargs(context)
      # TODO(klausg): move the get_state out of the try block
      # (after finding another way to make it compatible with TreeReduce)
      return self.metric.get_state(**kwargs)
    except KeyError:
      return self.metric.empty()

  def empty(self) -> Metric.State:
    return self.metric.empty()

  # Forwards `__kontext_keys__` so the keys can be extracted from the top-level
  def __kontext_keys__(self) -> dict[kontext.Key, kontext.Key]:
    return kontext.get_keypaths(self.metric)
