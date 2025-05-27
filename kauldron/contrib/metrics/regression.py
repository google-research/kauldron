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

"""Metrics for regression tasks."""

from __future__ import annotations

import dataclasses

import flax.struct
from jax import numpy as jnp
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RootMeanSquaredError(base.Metric):
  r"""Root Mean Squared Error.

  Compute the root mean squared error between the predictions and the targets.
  RMSE is a common metric for regression tasks which has different definitions
  depending on the library.

  If rmse_mode is 'average_all', we return the square root of average of
  squared errors. This implementation matches
  tf.keras.metrics.RootMeanSquaredError.
  .. math::
     \text{RMSE} =  \sqrt{\text{mean}((y_{\text{pred}} - y_{\text{target}})^2)}

  If rmse_mode is 'trad', we compute the square root of the mean of the
  squared the differences between the predictions and the targets. This matches
  https://en.wikipedia.org/wiki/Root_mean_square_deviation.
  .. math::
     \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \|pred_{i} - target_{i}\|^2}

  If rmse_mode is 'per_sample_sqrt', we compute the mean of the square roots of
  the sum of squared differences between the predictions and the targets.
  This is used
  .. math::
      \text{RMSE} = \frac{1}{N} \sum_{i=1}^N \sqrt{\|pred_{i} - target_{i}\|^2}

  Attributes:
    preds: The predictions.
    targets: The targets.
    rmse_mode: Different RMSE conventions are used across libraries. One of
      'trad', 'per_sample_sqrt' or 'average_all'.
  """

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED

  rmse_mode: str = "trad"

  @flax.struct.dataclass
  class State(base_state.CollectingState["RootMeanSquaredError"]):
    """RootMeanSquaredError state."""

    sum_squared_diffs: Float["b 1"]
    number_summed_values: Int["b"]

    @typechecked
    def compute(self) -> float:
      out = super().compute()
      sum_squared_diffs = out.sum_squared_diffs
      number_summed_values = out.number_summed_values

      if self.parent.rmse_mode == "trad":
        error = jnp.sqrt(jnp.mean(sum_squared_diffs))
      elif self.parent.rmse_mode == "average_all":
        error = jnp.sqrt(
            jnp.sum(sum_squared_diffs) / jnp.sum(number_summed_values)
        )
      elif self.parent.rmse_mode == "per_sample_sqrt":
        per_sample_sqrts = jnp.sqrt(sum_squared_diffs)
        error = jnp.mean(per_sample_sqrts)
      else:
        raise ValueError(
            f"Unknown rmse_mode: {self.parent.rmse_mode} not in ['trad',"
            " 'average_all', 'per_sample_sqrt']"
        )
      return float(error)

  @typechecked
  def get_state(
      self,
      preds: Float["b *d"],
      targets: Float["b *d"],
  ) -> RootMeanSquaredError.State:
    num_samples = targets.shape[0]
    square_diffs = jnp.square(preds - targets).reshape(num_samples, -1)
    num_values_per_sample = square_diffs.shape[1]
    sum_square_diffs = jnp.sum(square_diffs, axis=1)
    num_samples = targets.shape[0]
    # Collect for each sample the sum of squared errors along with the number
    # of values summed, so that we can compute the average later.
    # Collect only sums instead of the full arrays to save memory.
    return self.State(
        sum_squared_diffs=sum_square_diffs,
        number_summed_values=jnp.array(
            [num_values_per_sample for _ in range(num_samples)]
        ),
    )
