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

"""Metrics that keep simple statistics about generic values."""
from __future__ import annotations

import dataclasses
from typing import Optional

import flax.struct
import jax.numpy as jnp
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, Key, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class SingleDimension(base.Metric):
  """Returns a single chosen dimension of the tensor.

  Attributes:
    tensor: Key for the tensor to capture the value of.
    mask: Ignored.
    index: Dimension to index (from the last axis).
  """

  tensor: Key
  mask: Optional[Key] = None

  index: int = 0

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      tensor: Float["*any"],
  ) -> SingleDimension.State:
    value = tensor[..., self.index]
    return self.State.from_values(values=value)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Norm(base.Metric):
  """Wraps jnp.linalg.norm to compute the average norm for given tensors.

  Computes jnp.linalg.norm for the array corresponding to the "tensor" key, and
  averages the value over remaining dimensions (taking masking into account).

  See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html


  Attributes:
    tensor: Key for the tensor to compute the norm over.
    mask: Optional key for masking out some of the tensors (i.e. ignore them in
      the averaging).
    axis: Axis over which to compute the norm. If axis is an integer, it
      specifies the axis of x along which to compute the vector norms. If axis
      is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
      norms of these matrices are computed. If axis is None then either a vector
      norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The
      default is None.
    ord: Order of the norm. Possible values: None, "fro", "nuc", np.inf,
      -np.inf, -2, -1, 0, or any integer or float. See `np.linalg.norm`.
  """

  tensor: Key
  mask: Optional[Key] = None

  axis: None | int | tuple[int, int] = -1
  ord: float | int | str | None = None

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      tensor: Float["*any"],
      mask: Optional[Bool["*#any"]] = None,
  ) -> Norm.State:
    norm = jnp.linalg.norm(tensor, ord=self.ord, axis=self.axis, keepdims=True)

    if mask is not None:
      mask = jnp.broadcast_to(mask, norm.shape)

    # averaging of norms is done by the State
    return self.State.from_values(values=norm, mask=mask)


@flax.struct.dataclass
class StdState(base_state.State):
  """Computes the standard deviation of a scalar or a batch of scalars."""

  total: jnp.ndarray
  sum_of_squares: jnp.ndarray
  count: jnp.ndarray

  @classmethod
  def from_values(
      cls, values: jnp.ndarray, mask: jnp.ndarray | None = None, **_
  ) -> StdState:
    # Note: unlike clu.metrics.Std we support not just batches of scalars but
    # any shape of values. Thus we flatten the values before passing them on.
    values = jnp.ravel(values)
    mask = jnp.ravel(mask) if mask is not None else None

    if values.ndim == 0:
      values = values[None]
    assert values.ndim == 1
    if mask is None:
      mask = jnp.ones(values.shape[0], dtype=jnp.int32)
    return cls(
        total=jnp.where(mask, values, jnp.zeros_like(values)).sum(),
        sum_of_squares=jnp.where(
            mask, values**2, jnp.zeros_like(values)
        ).sum(),
        count=mask.sum(),
    )

  @classmethod
  def empty(cls) -> StdState:
    return cls(
        total=jnp.array(0, jnp.float32),
        sum_of_squares=jnp.array(0, jnp.float32),
        count=jnp.array(0, jnp.int32),
    )

  def merge(self, other: StdState) -> StdState:
    return type(self)(
        total=self.total + other.total,
        sum_of_squares=self.sum_of_squares + other.sum_of_squares,
        count=self.count + other.count,
    )

  def compute(self) -> jnp.ndarray:
    # var(X) = 1/N \sum_i (x_i - mean)^2
    #        = 1/N \sum_i (x_i^2 - 2 x_i mean + mean^2)
    #        = 1/N ( \sum_i x_i^2 - 2 mean \sum_i x_i + N * mean^2 )
    #        = 1/N ( \sum_i x_i^2 - 2 mean N mean + N * mean^2 )
    #        = 1/N ( \sum_i x_i^2 - N * mean^2 )
    #        = \sum_i x_i^2 / N - mean^2
    mean = self.total / self.count
    variance = self.sum_of_squares / self.count - mean**2
    # Mathematically variance can never be negative but in reality we may run
    # into such issues due to numeric reasons.
    variance = jnp.clip(variance, a_min=0.0)
    return variance**0.5


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Std(base.Metric):
  """Compute the standard deviation for float values.

  This is a simple example of wrapping a CLU metric.
  """

  values: Key
  mask: Optional[Key] = None

  @flax.struct.dataclass
  class State(StdState):
    pass

  @typechecked
  def get_state(
      self,
      values: Float["*b n"],
      mask: Optional[Float["*b 1"]] = None,
  ) -> Std.State:
    return self.State.from_values(values=values, mask=mask)
