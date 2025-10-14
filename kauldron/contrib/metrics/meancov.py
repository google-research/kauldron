# Copyright 2025 The kauldron Authors.
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

"""Metric state for tracking mean and covariance stats."""

from __future__ import annotations

import dataclasses
from typing import Self

import flax.struct
import jax.lax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


#
@flax.struct.dataclass
class MeanCov(kd.metrics.State):
  """Computes (possibly weighted) mean and covariance for given vectors.

  Uses a variant of Chan's method for parallel computation of covariance.
  See:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

  Tracks three quantities:
  * total: Sum of the vectors (i.e. sum(w * x, axis=0))
  * weight: Sum of the weights, or vector count. (i.e. sum(w) or x.shape[0])
  * squares: Sum of deviation products
    (i.e. sum(w * jnp.outer(x - mu, x - mu), axis=0))

  The mean and covariance are computed as follows:
  mean = total / weight
  cov = squares / weight  # biased covariance


  Usage:
  ```python

  class MyMetric(kd.metrics.Metric):
    ...

    @flax.struct.dataclass
    class State(kd.metrics.MeanCov):

      def compute(self):
        mean, cov = super().compute()
        return custom_metric(mean, cov)

    def get_state(self, samples: Float["b n"]) -> State:
      ...
      return self.State.from_samples(samples)
  ```


  References:
  [1] "Numerically Stable Parallel Computation of (Co-)Variance"
      Shubert et al. 2018
      https://ds.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
  """

  total: Float["n"] | None = None
  squares: Float["n n"] | None = None
  weight: Float[""] | None = None
  ddof: int = 0

  @classmethod
  def empty(cls) -> Self:
    return cls(total=None, squares=None, weight=None)

  @classmethod
  @typechecked
  def from_samples(
      cls,
      samples: Float["b n"],
      fweights: Int["b"] | None = None,
      # TODO(klausg): add support for aweights
      ddof: int = 0,
  ) -> "MeanCov":
    """Creates a MeanCov state from a batch of samples.

    Args:
      samples: A batch of samples (i.e. a collection of vectors)
      fweights: Optional 1-D array of integer frequency weights; the number of
        times each observation vector should be repeated. If None, all samples
        are assumed to have a weight of 1. Behaves the same as `fweights` in
        `np.cov()`.
      ddof: The normalization correction to compute the (un-)biased covariance.
        I.e. the covariance is computed as: `cov = squares / (weight - ddof)`.
        ddof=1 will return the unbiased covariance estimate, while ddof=0 will
        return the simple average of the squares.
    """
    if fweights is None:
      fweights = jnp.ones(samples.shape[:1], dtype=jnp.int32)
    weight = jnp.sum(fweights, dtype=jnp.float32)

    total = jnp.sum(fweights[:, None] * samples, axis=0)
    mean = total / weight
    diff = samples - mean
    squares = jnp.einsum(
        "b,bi,bj->ij",
        fweights,
        diff,
        diff,
        precision=jax.lax.Precision.HIGHEST,
    )
    return cls(total=total, squares=squares, weight=weight, ddof=ddof)

  def merge(self, other: Self) -> Self:  # pytype: disable=signature-mismatch
    if self.weight is None:
      return other
    elif other.weight is None:
      return self

    if self.ddof != other.ddof:
      raise ValueError("ddof must be the same for both states.")

    total = self.total + other.total
    weight = self.weight + other.weight

    diff = other.total * self.weight - self.total * other.weight
    squares = (
        self.squares
        + other.squares
        + jnp.outer(diff, diff) / (self.weight * other.weight * weight)
    )

    return dataclasses.replace(
        self, total=total, squares=squares, weight=weight
    )

  def finalize(self) -> Self:
    if self.total is None:
      return self
    # convert jax.Arrays to np.ndarrays
    return dataclasses.replace(
        self,
        total=np.array(self.total, dtype=np.float32),
        squares=np.array(self.squares, dtype=np.float32),
        weight=np.array(self.weight, dtype=np.float32),
    )

  def compute(self) -> tuple[Float["n"] | None, Float["n n"] | None]:
    if self.total is None:
      return None, None
    mean = self.total / self.weight

    if self.weight >= self.ddof:
      cov = self.squares / (self.weight - self.ddof)
    else:
      cov = np.full_like(self.squares, fill_value=np.nan)

    return mean, cov
