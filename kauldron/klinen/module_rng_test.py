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

"""Test."""

import chex
import flax
from flax import linen as nn
from jax import numpy as jnp
from kauldron import klinen as knn
from kauldron import random
import numpy as np


class AutoEncoder(knn.Module):
  encoder: nn.Module
  decoder: nn.Module

  @nn.compact
  def __call__(self, x):
    return self.decoder(self.encoder(x))


def test_init():
  key = random.PRNGKey(0)

  m0 = knn.Dense(3)
  m1 = nn.Dense(3)

  x = jnp.zeros((2,))

  p0 = m0.init_bind(key, x).params
  p1 = flax.core.freeze(m1.init(key, x)['params'])

  chex.assert_trees_all_close(p0, p1)  # klinen and linen have same params


def test_randomness():
  model = AutoEncoder(
      encoder=knn.Sequential([
          knn.Dropout(0.5),
          knn.Dense(32),
      ]),
      decoder=knn.Sequential([
          knn.Dropout(0.5),
          knn.Dense(32),
      ]),
  )

  key = random.PRNGKey(0)

  x = jnp.ones((5,))

  model = model.init_bind(key.fold_in('init'), x)

  model = model.with_rng({'dropout': key.fold_in('dropout')})

  # Calling the model directly or indirectly should yield the same result
  y0 = model(x)
  y1 = model.decoder(model.encoder(x))

  np.testing.assert_allclose(y0, y1)
