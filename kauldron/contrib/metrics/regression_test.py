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

"""Test for regression metrics."""

from kauldron.contrib.metrics import regression as regression_metrics
import numpy as np
import pytest


@pytest.mark.parametrize(
    ('targets', 'preds', 'expected_rmse', 'rmse_mode'),
    [
        # Reference example from tf.metrics.RootMeanSquaredError, which defines
        # RMSE with sqrt applied to the average of squared errors.
        (
            np.array([[0, 1], [0, 0]]),
            np.array([[1, 1], [0, 0]]),
            0.5,
            'average_all',
        ),
        # Reference example from
        # https://pytorch.org/ignite/v0.5.0.post2/generated/ignite.metrics.RootMeanSquaredError.html
        # which computes the average of sqrt errors computed separately for each
        # sample.
        (
            np.array([[1, 2, 4, 1], [2, 3, 1, 5], [1, 3, 5, 1], [1, 5, 1, 11]]),
            0.75
            * np.array(
                [[1, 2, 4, 1], [2, 3, 1, 5], [1, 3, 5, 1], [1, 5, 1, 11]]
            ),
            1.956559480312316,
            'trad',
        ),
    ],
)
def test_root_mean_squared_error(
    targets: np.ndarray,
    preds: np.ndarray,
    expected_rmse: float,
    rmse_mode: str,
):
  metric = regression_metrics.RootMeanSquaredError(rmse_mode=rmse_mode)
  metric_value = metric.get_state(
      preds=preds.astype(np.float32), targets=targets.astype(np.float32)
  ).compute()

  np.testing.assert_allclose(metric_value, expected_rmse, atol=1e-3)
