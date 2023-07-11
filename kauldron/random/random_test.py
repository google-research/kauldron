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

"""Tests."""

from flax import linen as nn
import jax
from kauldron import random
import numpy as np


def test_base():
  key = random.PRNGKey(0)
  val = key.uniform((5,))
  assert isinstance(val, jax.Array)
  assert val.shape == (5,)

  np.testing.assert_array_equal(
      key.uniform((5,)),
      jax.random.uniform(key, (5,)),
  )


def test_jit():
  @jax.jit
  def fn(key: random.PRNGKey):
    x0, x1 = key.split()
    return x0.normal(), x1

  key = random.PRNGKey(0)
  out, out_key = fn(key)
  assert isinstance(out, jax.Array)
  assert isinstance(out_key, random.PRNGKey)


def test_flax():
  class MLP(nn.Module):

    @nn.compact
    def __call__(self, x):
      return nn.Dense(features=10)(x)

  key = random.PRNGKey(0)

  model = MLP()
  variables = model.init(
      key.fold_in(1),
      key.normal((10,)),
  )  # Initialization call
  del variables


def test_getitem():
  key = random.PRNGKey(0).split(2)
  assert isinstance(key, random.PRNGKey)
  assert isinstance(key[0], random.PRNGKey)


def test_foldin():
  key = random.PRNGKey(0)
  key0 = key.fold_in('dropout')
  key1 = key.fold_in('dropout')

  assert isinstance(key0, random.PRNGKey)

  np.testing.assert_array_equal(key0, key1)


def test_tree_map():
  out = jax.tree_util.tree_map(lambda x: None, {'x': random.PRNGKey(0)})
  assert out == {'x': None}

  key = random.PRNGKey(0)
  key = key.split(3)
  assert isinstance(key, random.PRNGKey)
  key2 = jax.tree_util.tree_map(lambda x: x, key)
  np.testing.assert_array_equal(key.rng, key2.rng)

  np_key = np.asarray(key)
  np.testing.assert_allclose(np_key, key.rng)
