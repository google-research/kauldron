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

"""Tests."""

import jax
from kauldron import random
import numpy as np


def test_fold_in_str():
  key = jax.random.PRNGKey(0)
  k1 = random.fold_in_str(key, 'test')
  k2 = random.fold_in_str(key, 'test')
  k3 = random.fold_in_str(key, 'other')

  assert np.array_equal(k1, k2)
  assert not np.array_equal(k1, k3)


def test_random_seed():
  key = jax.random.PRNGKey(0)
  seed = random.random_seed(key)

  assert isinstance(seed, int)
  assert 0 <= seed < 2**32

  key2 = jax.random.PRNGKey(1)
  seed2 = random.random_seed(key2)
  assert seed != seed2


def test_prngkey_compatibility():
  # 1. Verify JAX Fry key wrapping behaves correctly
  if hasattr(jax.random, 'key'):
    # Modern JAX versions (>=0.4.16) have jax.random.key
    fry_key = jax.random.key(42)
    wrapped_fry_key = random.PRNGKey(fry_key)
    # The wrapped key's internal rng should be the same Fry key array
    assert np.array_equal(
        jax.random.key_data(wrapped_fry_key.rng), jax.random.key_data(fry_key)
    )

  # 2. Verify dynamic OO API delegation behaves correctly
  key = random.PRNGKey(42)

  # Test split() delegation
  k1, k2 = key.split()
  assert isinstance(k1, random.PRNGKey)
  assert isinstance(k2, random.PRNGKey)

  # Test uniform() delegation (which is in jax.random)
  val = key.uniform()
  assert not val.shape
  assert 0.0 <= val <= 1.0
