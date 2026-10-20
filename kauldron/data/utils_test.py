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

"""Tests for `kauldron.data.utils`."""

from etils import enp
import jax
import jax.numpy as jnp
from kauldron.data import utils as data_utils
import numpy as np


def _primitives(fn) -> set[str]:
  return {eqn.primitive.name for eqn in jax.make_jaxpr(fn)().jaxpr.eqns}


def test_mock_array_is_not_an_uninitialized_buffer():
  # `jnp.empty` lowers to an `AllocateBuffer` custom call on GPU and TPU, which
  # the SPMD partitioner does not shard: the mock batch was allocated whole on
  # every device before its sharding constraint applied.
  spec = enp.ArraySpec(shape=(None, 4, 3), dtype=np.float64)
  fn = lambda: data_utils.array_spec_to_jnp_empty(spec, batch_dim=2)
  assert 'empty' not in _primitives(fn)

  mock = fn()
  assert mock.shape == (2, 4, 3)
  assert mock.dtype == jnp.float32


def test_mock_array_makes_dynamic_dims_concrete():
  spec = enp.ArraySpec(shape=(1, None, None), dtype=np.int64)
  fn = lambda: data_utils.array_spec_to_jnp_empty(spec)
  assert 'empty' not in _primitives(fn)

  mock = fn()
  assert mock.shape == (1, 1, 1)
  assert mock.dtype == jnp.int32
