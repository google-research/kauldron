# Copyright 2026 The kauldron Authors.
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

"""Regression tests for weighted standard-deviation states."""

import jax
import jax.numpy as jnp
from kauldron.metrics import stats
import numpy as np
import pytest


def reference(values, weights):
  values = np.asarray(values, dtype=np.float64)
  weights = np.broadcast_to(np.asarray(weights, dtype=np.float64), values.shape)
  selected = weights != 0
  mean = np.average(values[selected], weights=weights[selected])
  return np.sqrt(
      np.average((values[selected] - mean) ** 2, weights=weights[selected])
  )


@pytest.mark.parametrize(
    'weights',
    [
        [0.5, 0.5, 0.5],
        [0.25, 0.5, 1.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.5, 2.0],
        [True, False, True],
        1.0,
        0.25,
    ],
)
def test_fractional_masks_are_weights_for_both_moments(weights):
  values = jnp.array([1.0, 3.0, 6.0])
  mask = jnp.asarray(weights)
  state = stats.StdState.from_values(values, mask=mask)
  expected = reference(values, weights)
  np.testing.assert_allclose(state.compute(), expected, rtol=3e-6, atol=1e-6)
  broadcast = np.broadcast_to(np.asarray(weights), values.shape)
  np.testing.assert_allclose(
      state.total, np.sum(np.asarray(values) * broadcast)
  )
  np.testing.assert_allclose(
      state.sum_of_squares, np.sum(np.asarray(values) ** 2 * broadcast)
  )
  np.testing.assert_allclose(state.count, broadcast.sum())


@pytest.mark.parametrize('factor', [0.125, 2.0, 10.0])
def test_rescaling_all_weights_does_not_change_standard_deviation(factor):
  values = jnp.array([1.0, 3.0, 6.0])
  weights = jnp.array([1.0, 2.0, 3.0])
  actual = stats.StdState.from_values(values, weights * factor).compute()
  expected = reference(values, weights)
  np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-6)


def test_merged_batches_match_weighted_full_population():
  values = jnp.array([1.0, 3.0, 6.0, 9.0, 11.0])
  weights = jnp.array([0.25, 0.5, 1.0, 2.0, 0.75])
  first = stats.StdState.from_values(values[:2], weights[:2])
  second = stats.StdState.from_values(values[2:], weights[2:])
  merged = stats.StdState.empty().merge(first).merge(second)
  np.testing.assert_allclose(
      merged.compute(), reference(values, weights), rtol=3e-6
  )
  np.testing.assert_allclose(
      merged.compute(), second.merge(first).compute(), rtol=0, atol=0
  )


def test_broadcast_masks_work_through_the_public_metric():
  values = jnp.array([[1.0, 2.0], [3.0, 4.0], [7.0, 8.0]])
  weights = jnp.array([[0.25], [0.5], [2.0]])
  metric = stats.Std(values='x')
  expected = reference(values, weights)
  for fn in (
      lambda x, w: metric.get_state(values=x, mask=w).compute(),
      jax.jit(lambda x, w: metric.get_state(values=x, mask=w).compute()),
  ):
    np.testing.assert_allclose(fn(values, weights), expected, rtol=3e-6)


def test_zero_weight_nonfinite_values_remain_excluded():
  values = jnp.array([1.0, 3.0, jnp.nan, jnp.inf])
  weights = jnp.array([0.5, 0.5, 0.0, 0.0])
  state = stats.StdState.from_values(values, weights)
  np.testing.assert_allclose(state.compute(), 1.0, rtol=0, atol=0)


def test_absent_and_boolean_masks_keep_existing_results():
  values = jnp.array([1.0, 3.0, 6.0])
  np.testing.assert_allclose(
      stats.StdState.from_values(values).compute(), np.std(values), rtol=2e-6
  )
  np.testing.assert_allclose(
      stats.StdState.from_values(
          values, jnp.array([True, False, True])
      ).compute(),
      2.5,
  )
  assert np.isnan(stats.StdState.from_values(values, jnp.zeros(3)).compute())