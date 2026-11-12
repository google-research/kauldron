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

from etils import enp
import jax
import jax.numpy as jnp
from kauldron.data import utils as data_utils
from kauldron.utils import sharding_utils
import numpy as np


def _primitives(fn) -> set[str]:
  return {eqn.primitive.name for eqn in jax.make_jaxpr(fn)().jaxpr.eqns}


def test_array_spec_to_jnp_zeros_is_not_an_uninitialized_buffer():
  # `jnp.empty` lowers to an `AllocateBuffer` custom call on GPU and TPU, which
  # the SPMD partitioner does not shard: the mock batch was allocated whole on
  # every device before its sharding constraint applied.
  spec = enp.ArraySpec(shape=(None, 4, 3), dtype=np.float64)
  fn = lambda: data_utils.array_spec_to_jnp_zeros(spec, batch_dim=2)
  assert 'empty' not in _primitives(fn)

  mock = fn()
  assert mock.shape == (2, 4, 3)
  assert mock.dtype == jnp.float32


def test_array_spec_to_jnp_zeros_makes_dynamic_dims_concrete():
  spec = enp.ArraySpec(shape=(1, None, None), dtype=np.int64)
  fn = lambda: data_utils.array_spec_to_jnp_zeros(spec)
  assert 'empty' not in _primitives(fn)

  mock = fn()
  assert mock.shape == (1, 1, 1)
  assert mock.dtype == jnp.int32


def test_array_spec_to_jnp_empty_alias_and_mock_batch():
  assert (
      data_utils.array_spec_to_jnp_empty is data_utils.array_spec_to_jnp_zeros
  )

  elem_spec = {'x': enp.ArraySpec(shape=(2, 4), dtype=np.float32)}
  batch = data_utils.mock_batch_from_elem_spec(
      elem_spec, sharding_utils.sharding.REPLICATED
  )
  assert batch['x'].shape == (2, 4)
  assert batch['x'].dtype == jnp.float32
  np.testing.assert_array_equal(batch['x'], np.zeros((2, 4), dtype=np.float32))
