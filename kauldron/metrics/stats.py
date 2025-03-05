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

"""Metrics that keep simple statistics about generic values."""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional

import flax.struct
import jax.numpy as jnp
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class SingleDimension(base.Metric):
  """Returns a single chosen dimension of the tensor.

  Attributes:
    tensor: kontext.Key for the tensor to capture the value of.
    index: Dimension to index (from the last axis). If None, no indexing is
      performed.
  """

  tensor: kontext.Key = kontext.REQUIRED

  index: int | None = 0

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(self, tensor: Float["*any"]) -> SingleDimension.State:
    if self.index is not None:
      tensor = tensor[..., self.index]
    return self.State.from_values(values=tensor)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Norm(base.Metric):
  """Wraps jnp.linalg.norm to compute the average norm for given tensors.

  Computes jnp.linalg.norm for the array corresponding to the "tensor" key, and
  averages the value over remaining dimensions (taking masking into account).

  See: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html


  Attributes:
    tensor: kontext.Key for the tensor to compute the norm over.
    mask: Optional key for masking out some of the tensors (i.e. ignore them in
      the averaging).
    axis: Axis over which to compute the norm. If axis is an integer, it
      specifies the axis of x along which to compute the vector norms. If axis
      is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
      norms of these matrices are computed. If axis is None then either a vector
      norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned.
    ord: Order of the norm. Possible values: None, "fro", "nuc", np.inf,
      -np.inf, -2, -1, 0, or any integer or float. See `np.linalg.norm`.
    aggregation_type: How to aggregate the norms in TreeReduce. Average will
      compute the average of the norms. Concat will compute the norm as if all
      nodes of a tree were concatenated into a single vector. Average by
      default.
  """

  tensor: kontext.Key = kontext.REQUIRED
  mask: Optional[kontext.Key] = None

  axis: None | int | tuple[int, int] = -1
  ord: float | int | None = None

  aggregation_type: Literal["average", "concat"] | None = None

  @flax.struct.dataclass
  class State(base_state.AverageState["Norm"]):
    """Wrapper around AverageState for Norm."""

    def merge(self, other: base_state.AverageState) -> base_state.AverageState:
      assert isinstance(other, Norm.State)

      if self.parent.axis is None and self.parent.aggregation_type is None:
        status.warn(
            "When setting axis=None in kd.metrics.Norm and running a TreeReduce"
            " over it, Norm will average the norms of individual leaves, rather"
            " than computing the norm as if everything was concatenated. Please"
            ' specify an aggregation_type to "concat" to get the norm of'
            ' concatenated values. Set the aggregation_type to "average" to'
            " suppress this warning."
        )
      return super().merge(other)

    def compute(self) -> Float[""]:
      parent = self.parent
      aggregation_type = (
          parent.aggregation_type
          if parent.aggregation_type is not None
          else "average"
      )
      if aggregation_type == "average":
        return super().compute()
      elif aggregation_type == "concat":
        # norm(v) = (\sum_i |v_i|**ord)**(1/ord)
        # Note that there is no averaging here, as expected by the formula.
        return self.total ** (1 / (parent.ord or 2))
      raise ValueError(
          f"Unsupported aggregation_type: {parent.aggregation_type}"
      )

  @typechecked
  def get_state(
      self,
      tensor: Float["*any"],
      mask: Optional[Bool["*#any"] | Float["*#any"]] = None,
  ) -> Norm.State:
    if self.ord is not None and self.axis is None:
      # self.ord is None and self.axis is None is a special case that calls
      # .ravel()
      # If we need ord, but axis is None, we are going to call ravel()
      # ourselves.
      norm = jnp.linalg.norm(tensor.ravel(), ord=self.ord, keepdims=False)
      # manual keepdims
      norm = norm.reshape((1,) * tensor.ndim)
    else:
      norm = jnp.linalg.norm(
          tensor, ord=self.ord, axis=self.axis, keepdims=True
      )

    if mask is not None:
      mask = jnp.broadcast_to(mask, norm.shape)

    aggregation_type = (
        self.aggregation_type
        if self.aggregation_type is not None
        else "average"
    )
    values_for_averaging = (
        norm if aggregation_type == "average" else norm ** (self.ord or 2)
    )
    return self.State.from_values(
        values=values_for_averaging,
        mask=mask,
    )


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
    if mask is None:
      mask = jnp.ones([])
    mask = jnp.broadcast_to(mask, values.shape)
    return cls(
        total=jnp.where(mask, values, jnp.zeros_like(values)).sum(),
        sum_of_squares=jnp.where(mask, values**2, jnp.zeros_like(values)).sum(),
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
    variance = jnp.clip(variance, min=0.0)
    return variance**0.5


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Std(base.Metric):
  """Compute the standard deviation for float values."""

  values: kontext.Key = kontext.REQUIRED
  mask: Optional[kontext.Key] = None

  @flax.struct.dataclass
  class State(StdState):
    pass

  @typechecked
  def get_state(
      self,
      values: Float["*b n"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> Std.State:
    return self.State.from_values(values=values, mask=mask)
