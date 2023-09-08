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

"""Base state."""

from __future__ import annotations

import abc
import dataclasses
import types
from typing import Any, TypeVar

from etils import epy
import flax
import jax
import jax.numpy as jnp
from kauldron.typing import Float
import numpy as np

_SelfT = TypeVar("_SelfT")


class State(abc.ABC):
  """Base metric state class.

  In Kauldron, `kd.metrics.Metric` are stateless pure-python objects. Instead,
  each metrics emit a `kd.metrics.State` when calling `state = metric(**kwargs)`
  (often inside the `jax.jit` train or eval step).

  Those states can then be accumulated across multiple steps (with
  `state.merge`) before computing the final value (with `metric.compute(state)`)

  ```python
  metric = kd.metric.Accuracy()

  state = metric(logits=logits, labels=labels)

  # Optionally accumulate the state across multiple batches
  state = state.merge(other_state)

  values = metric.compute(state)  # Get the final value
  ```
  """

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


@flax.struct.dataclass
class CollectingState(State):
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
  # TODO(epot): Could add a more explicit annotation: `x: Accumulated[Float[]]`
  # so a metric can have both accumulated and reduced fields

  def __post_init__(self):
    # Normalize array values to `tuple()`
    for k, val in self._accumulated_fields.items():
      if not isinstance(val, tuple):  # Normalize `array` to tuple
        val = jnp.array(val)
        if val.shape == ():  # Scalars are broadcasted (for concatenation)  # pylint: disable=g-explicit-bool-comparison
          val = val[None, ...]
        object.__setattr__(self, k, (val,))

  @property
  def _accumulated_fields(self):
    return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

  @classmethod
  def empty(cls: type[_SelfT]) -> _SelfT:
    # Only empty tuples
    return cls(**{f.name: () for f in dataclasses.fields(cls)})

  def merge(self: _SelfT, other: _SelfT) -> _SelfT:
    merged_fields = {
        k: _merge_normalize_tuple(v1, v2)
        for k, (v1, v2) in epy.zip_dict(
            self._accumulated_fields, other._accumulated_fields  # pylint: disable=protected-access
        )
    }
    return type(self)(**merged_fields)

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
  return jax.tree_map(np.asarray, values)


# Inherit for better tracability/debug messages (so user can search and find
# this class)
class _CollectingStateOutput(types.SimpleNamespace):
  pass


# TODO(epot): Could be unified with `AllReduceMean` in `kd.losses`
@flax.struct.dataclass
class AverageState(State):
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
      values: jnp.ndarray,
      *,
      mask: jnp.ndarray | None = None,
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
    mask = mask.astype(bool)
    return cls(
        total=jnp.where(mask, values, jnp.zeros_like(values)).sum(),
        count=jnp.where(
            mask,
            jnp.ones_like(values, dtype=jnp.int32),
            jnp.zeros_like(values, dtype=jnp.int32),
        ).sum(),
    )

  @classmethod
  def empty(cls) -> AverageState:
    return cls(
        total=jnp.array(0.0, jnp.float32),
        count=jnp.array(0.0, jnp.int32),
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
    return self.total / self.count
