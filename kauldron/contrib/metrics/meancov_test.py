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

"""Tests for MeanCov metric."""

import jax.numpy as jnp
from kauldron.contrib.metrics import meancov
import numpy as np
import pytest


@pytest.mark.parametrize('ddof', [0, 1])
def test_meancov_unweighted(ddof):
  # Create sample data
  samples1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  samples2 = np.array([[7.0, 8.0], [9.0, 10.0]])
  all_samples = np.concatenate([samples1, samples2], axis=0)

  # Compute mean and cov directly
  mean_np = np.mean(all_samples, axis=0)
  cov_np = np.cov(all_samples, rowvar=False, ddof=ddof)

  # Compute mean and cov with MeanCov
  state1 = meancov.MeanCov.from_samples(samples=jnp.array(samples1), ddof=ddof)
  state2 = meancov.MeanCov.from_samples(samples=jnp.array(samples2), ddof=ddof)
  merged_state = state1.merge(state2)
  mean_mc, cov_mc = merged_state.compute()

  np.testing.assert_allclose(mean_mc, mean_np, atol=1e-6)
  np.testing.assert_allclose(cov_mc, cov_np, atol=1e-6)


@pytest.mark.parametrize('ddof', [0, 1])
def test_meancov_weighted(ddof):
  # Create sample data
  samples1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  weights1 = np.array([1, 2, 2], dtype=np.int32)
  samples2 = np.array([[7.0, 8.0], [9.0, 10.0]])
  weights2 = np.array([3, 4], dtype=np.int32)
  all_samples = np.concatenate([samples1, samples2], axis=0)
  all_weights = np.concatenate([weights1, weights2], axis=0)

  # Compute mean and cov directly
  mean_np = np.average(all_samples, axis=0, weights=all_weights)
  cov_np = np.cov(all_samples, rowvar=False, ddof=ddof, fweights=all_weights)

  # Compute mean and cov with MeanCov
  state1 = meancov.MeanCov.from_samples(
      samples=jnp.array(samples1), fweights=jnp.array(weights1), ddof=ddof
  )
  state2 = meancov.MeanCov.from_samples(
      samples=jnp.array(samples2), fweights=jnp.array(weights2), ddof=ddof
  )
  merged_state = state1.merge(state2)
  mean_mc, cov_mc = merged_state.compute()

  np.testing.assert_allclose(mean_mc, mean_np, atol=1e-6)
  np.testing.assert_allclose(cov_mc, cov_np, atol=1e-6)


def test_meancov_unweighted_single_batch():
  samples = np.array(
      [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
  )
  mean_np = np.mean(samples, axis=0)
  cov_np = np.cov(samples, rowvar=False, ddof=0)

  state = meancov.MeanCov.from_samples(samples=jnp.array(samples))
  mean_mc, cov_mc = state.compute()

  np.testing.assert_allclose(mean_mc, mean_np, atol=1e-6)
  np.testing.assert_allclose(cov_mc, cov_np, atol=1e-6)


def test_meancov_weighted_single_batch():
  samples = np.array(
      [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
  )
  weights = np.array([1, 2, 1, 2, 3], dtype=np.int32)
  mean_np = np.average(samples, axis=0, weights=weights)
  cov_np = np.cov(samples, rowvar=False, ddof=0, aweights=weights)

  state = meancov.MeanCov.from_samples(
      samples=jnp.array(samples), fweights=jnp.array(weights)
  )
  mean_mc, cov_mc = state.compute()

  np.testing.assert_allclose(mean_mc, mean_np, atol=1e-6)
  np.testing.assert_allclose(cov_mc, cov_np, atol=1e-6)
