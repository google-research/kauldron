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

"""Base state."""

from __future__ import annotations

import abc
import dataclasses
import enum
import functools
import types
import typing
from typing import Any, Generic, TypeVar

from etils import epy
import flax.struct
import jax
import jax.numpy as jnp
from kauldron.typing import Array, Bool, Float  # pylint: disable=g-multiple-import
import numpy as np


if typing.TYPE_CHECKING:
  from kauldron.metrics import base  # pylint: disable=g-bad-import-order

  _MetricT = TypeVar("_MetricT", bound=base.Metric)
else:
  _MetricT = TypeVar("_MetricT")
_SelfT = TypeVar("_SelfT")
_FnT = TypeVar("_FnT")


# define _EMPTY_TYPE as an enum, which allows us to use
# Literal[_EMPTY_TYPE.EMPTY].
# See also dataclasses.MISSING and
# https://github.com/python/typeshed/pull/5900#issuecomment-895513797
class _EMPTY_TYPE(enum.Enum):  # pylint: disable=invalid-name
  """Sentinel value to indicate an empty field (e.g. parent)."""

  EMPTY = enum.auto()


EMPTY = _EMPTY_TYPE.EMPTY


@flax.struct.dataclass
class State(abc.ABC, Generic[_MetricT]):
  """Base metric state class.

  In Kauldron, `kd.metrics.Metric` are stateless pure-python objects. Instead,
  each metrics emit a `kd.metrics.State` when calling `state = metric(**kwargs)`
  (often inside the `jax.jit` train or eval step).

  Those states can then be accumulated across multiple steps (with
  `state.merge`) before computing the final value (with `state.compute()`)

  ```python
  metric = kd.metric.Accuracy()

  state = metric.get_state(logits=logits, labels=labels)

  # Optionally accumulate the state across multiple batches
  state = state.merge(other_state)

  values = state.compute()  # Get the final value
  ```

  Attribute:
    parent: A reference to the metric that emitted this state. Automatically
      added by `metric.get_state()`.
  """

  _: dataclasses.KW_ONLY
  parent: _MetricT = flax.struct.field(
      pytree_node=False, default=EMPTY
  )  # pytype: disable=annotation-type-mismatch

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # TODO(epot): Could also check that the 2 states are merged from the
    # same metric !!
    cls.merge = _propagate_parent_in_merge(cls.merge)

  @classmethod
  @abc.abstractmethod
  def empty(cls: type[_SelfT]) -> _SelfT:
    """Returns an empty instance (i.e. `.merge(State.empty())` is a no-op)."""
    raise NotImplementedError("Abstract method.")

  @abc.abstractmethod
  def merge(self: _SelfT, other: _SelfT) -> _SelfT:
    """Returns a new state that is the accumulation of `self` and `other`.

    Args:
      other: A `State` whose intermediate values should be accumulated onto the
        values of `self`.

    Returns:
      A new `State` that accumulates the value from both `self` and `other`.
    """
    raise NotImplementedError("Abstract method.")

  @abc.abstractmethod
  def compute(self) -> Any:
    """Computes final metrics from intermediate values."""
    raise NotImplementedError("Abstract method.")

  @staticmethod
  def isinstance(other) -> bool:
    """Returns whether `other` is a `State` (used in `tree_map(is_leaf=)`)."""
    return isinstance(other, State)


def _propagate_parent_in_merge(old_merge: _FnT) -> _FnT:
  """Propagate the parent."""

  @functools.wraps(old_merge)
  def new_merge(self, other):
    # TODO(epot): Comparison should ignore the `key: kontext.Key` (valid to
    # merge metrics from 2 differents origins)

    if self.parent != other.parent and not (
        self.parent is EMPTY or other.parent is EMPTY
    ):
      raise ValueError(
          "Trying to merge state comming from different metrics:"
          f" {self.parent} != {other.parent}\n"
          "If this is raised because the kontext.Keys are differents, you can "
          "open an issue."
      )
    parent = self.parent if self.parent is not EMPTY else other.parent
    new_self = old_merge(self, other)
    new_self = dataclasses.replace(new_self, parent=parent)
    return new_self

  return new_merge


@flax.struct.dataclass
class EmptyState(State[_MetricT]):
  """Empty state."""

  @classmethod
  def empty(cls) -> EmptyState:
    return cls()

  def merge(self, other: EmptyState) -> EmptyState:
    return self

  def compute(self) -> dict[Any, Any]:
    return {}


@flax.struct.dataclass
class CollectingState(State[_MetricT]):
  """Accumulate outputs across multiple steps (without reducing).

  Example:

  ```python
  @flax.struct.dataclass
  class AveragePrecision(kd.metrics.CollectingState):
    labels: Float['n_samples']
    logits: Float['n_samples n_classes']

    def compute(self):
      values = super().compute()  # Concatenate all accumulated values
      return sklearn.metrics.average_precision_score(  # Reduce
          values.labels,
          values.logits,
      )

  state0 = AveragePrecision(labels=labels0, logits=logits0)
  state1 = AveragePrecision(labels=labels1, logits=logits1)

  final_state = state0.merge(state1)  # Accumulate the values

  out = final_state.compute()  # Concatenate and reduce
  ```

  *  Internally, the states are normalized and stored as tuple:
     * `state0.labels = (labels0,))`
     * `final_state.labels = (labels0, labels1))`
  *  `state.merge(other_state)` only accumulate the values (append to a tuple).
  *  Reduction is only applied on `.compute()`
  *  To support mask, the subclass can accumulate the mask values and use it in
     the final computation.
  *  Because `merge()` keep all values, those metrics uses much more memory and
     can be slow to compute.
  """

  def __post_init__(self):
    # Normalize array values to `tuple()`
    for k, val in self._accumulated_fields.items():
      if not isinstance(val, tuple):  # Normalize `array` to tuple
        if not isinstance(val, (int, float, np.ndarray, jax.Array)):
          raise TypeError(
              f"Collecting state got non-array input: {k}={val}.\n"
              "Please open an issue if you need your metric state to support "
              "non-array attributes."
          )
        val = jnp.asarray(val)
        if val.shape == ():  # Scalars are broadcasted (for concatenation)  # pylint: disable=g-explicit-bool-comparison
          val = val[None, ...]
        object.__setattr__(self, k, (val,))

  @property
  def _accumulated_fields(self) -> dict[str, Array]:
    return {
        f.name: getattr(self, f.name)
        for f in dataclasses.fields(self)
        if f.name != "parent"  # Do not process `state.parent`
    }

  @classmethod
  def empty(cls: type[_SelfT]) -> _SelfT:
    # Only empty tuples
    return cls(
        **{f.name: () for f in dataclasses.fields(cls) if f.name != "parent"}
    )

  def merge(self: _SelfT, other: _SelfT) -> _SelfT:
    merged_fields = {
        k: _merge_normalize_tuple(v1, v2)
        for k, (v1, v2) in epy.zip_dict(
            self._accumulated_fields, other._accumulated_fields  # pylint: disable=protected-access,attribute-error
        )
    }
    return dataclasses.replace(self, **merged_fields)

  # Return `_SeltT` so auto-complete work
  def compute(self: _SelfT) -> _SelfT:
    """Returns the concatenated values."""
    return _CollectingStateOutput(  # pytype: disable=bad-return-type
        **{k: np.concatenate(v) for k, v in self._accumulated_fields.items()}  # pylint: disable=protected-access
    )


def _merge_normalize_tuple(v0, v1):
  assert isinstance(v0, tuple)
  assert isinstance(v1, tuple)
  values = v0 + v1
  if any(isinstance(v, jax.core.Tracer) for v in values):
    raise RuntimeError(
        "Tracer detected! CollectingState.merge should not be JIT compiled."
    )
  # TODO(epot): Should be executed asynchronously (blocking)
  return tuple(np.asarray(v) for v in values)


# Inherit for better tracability/debug messages (so user can search and find
# this class)
class _CollectingStateOutput(types.SimpleNamespace):
  pass


# TODO(epot): Could be unified with `AllReduceMean` in `kd.losses`
@flax.struct.dataclass
class AverageState(State[_MetricT]):
  """Computes the average of a scalar or a batch of tensors.

  Supports the following types of masks:

  - A one-dimensional mask with the same leading dimension as the scalars, or,
  - A multi-dimensional mask with the exact same dimensions as the scalars.
    This allows the use of per-example masks for examples in a batch, as well as
    per-target masks for targets for examples in a batch.

  The result is always a scalar.
  """

  total: Float[""]
  count: Float[""]

  @classmethod
  def from_values(
      cls,
      values: Float["b *any"],
      *,
      mask: Bool["b *#any"] | Float["b *#any"] | None = None,
  ) -> AverageState:
    """Factory to create the state from an array."""
    if values.ndim == 0:
      values = values[None]
    if mask is None:
      mask = jnp.ones_like(values)
    # Leading dimensions of mask and values must match.
    if mask.shape[0] != values.shape[0]:
      raise ValueError(
          "Argument `mask` must have the same leading dimension as `values`. "
          f"Received mask of dimension {mask.shape} "
          f"and values of dimension {values.shape}."
      )
    # Broadcast mask to the same number of dimensions as values.
    if mask.ndim < values.ndim:
      mask = jnp.expand_dims(
          mask, axis=tuple(np.arange(mask.ndim, values.ndim))
      )
    return cls(
        total=(values * mask).sum(),
        count=jnp.broadcast_to(mask, values.shape).sum(dtype=jnp.float32),
    )

  @classmethod
  def empty(cls) -> AverageState:
    return cls(
        total=jnp.array(0.0, jnp.float32),
        count=jnp.array(0.0, jnp.float32),
    )

  def merge(self, other: AverageState) -> AverageState:
    if self.total.shape != other.total.shape:
      raise ValueError(
          f"Expected same shape: {self.total.shape} != {other.total.shape}"
      )
    return type(self)(
        total=self.total + other.total,
        count=self.count + other.count,
    )

  def compute(self) -> Float[""]:
    if self.count == 0: return 0.0

    return self.total / self.count


@flax.struct.dataclass
class CollectFirstState(State[_MetricT]):
  """Get the first outputs (possibly) across multiple steps (no reducing).

  Example:

  ```python
  @flax.struct.dataclass
  class FirstNImages(kd.metrics.CollectFirstState):
    images: Float['N h w 3']


  state0 = FirstNImages(images=jnp.zeros((4, 16, 16, 3)), keep_first=5)
  state1 = FirstNImages(images=jnp.ones((4, 16, 16, 3)), keep_first=5)
  final_state = state0.merge(state1)
  assert final_state.compute().images.shape == (5, 16, 16, 3)
  ```
  """

  _: dataclasses.KW_ONLY
  # TODO(klausg): support None for unlimited collection?
  # TODO(klausg): Unify with CollectingState?
  keep_first: int = flax.struct.field(pytree_node=False)

  # TODO(klausg) dynamically check type annotations in post_init

  _INTERNAL_FIELDS = {"keep_first", "parent"}

  @classmethod
  def empty(cls: type[_SelfT]) -> _SelfT:
    return cls(**{
        f.name: None
        for f in dataclasses.fields(cls)
        if f.name not in cls._INTERNAL_FIELDS
    })

  @property
  def _accumulated_fields(self) -> dict[str, Array]:
    return {
        f.name: _maybe_truncate(getattr(self, f.name), self.keep_first)
        for f in dataclasses.fields(self)
        if f.name not in self._INTERNAL_FIELDS
    }

  def merge(self: _SelfT, other: _SelfT) -> _SelfT:
    assert hasattr(other, "keep_first")
    if self.keep_first != other.keep_first:
      raise ValueError(
          f"Expected same keep_first: {self.keep_first} != {other.keep_first}"
      )
    merged_fields = {
        k: _concat_truncate(v1, v2, self.keep_first)  # merge & truncate
        for k, (v1, v2) in epy.zip_dict(
            self._accumulated_fields, other._accumulated_fields  # pylint: disable=protected-access,attribute-error
        )
    }
    return dataclasses.replace(self, **merged_fields)

  # Return `_SelfT` so auto-complete works
  def compute(self: _SelfT) -> _SelfT:
    """Returns the concatenated values."""
    return _CollectingStateOutput(**self._accumulated_fields)  # pytype: disable=bad-return-type


def _maybe_truncate(v: Array["b *any"] | None, num: int):
  if v is None:
    return None
  return np.asarray(v[:num])


def _concat_truncate(v1, v2, num: int):
  """Concatenate two arrays along dim 0 up to length num_samples."""
  if v1 is None:
    return _maybe_truncate(v2, num)

  if v1.shape[0] < num and v2 is not None:
    n = num - v1.shape[0]
    v1 = np.concatenate([v1, v2[:n]], axis=0)

  if isinstance(v1, jax.core.Tracer):
    raise RuntimeError(
        "Tracer detected! CollectingState.merge should not be JIT compiled."
    )

  return _maybe_truncate(v1, num)
