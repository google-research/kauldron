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

from jax import numpy as jnp
from kauldron.metrics import base
from kauldron.metrics import stats
import numpy as np
import pytest


def test_norm_default():
  a = jnp.array(
      [
          [1, 1, 1],
          [2, 2, 2],
      ],
      dtype=jnp.float32,
  )
  b = jnp.array(
      [
          [3, 3, 3],
          [4, 4, 4],
      ],
      dtype=jnp.float32,
  )
  ab = jnp.asarray([a, b])

  full_tree = {"a": a, "b": b}
  concat_norm_reduce = base.TreeReduce(metric=stats.Norm(tensor="full_tree"))
  value = concat_norm_reduce.get_state(tensor=full_tree).compute()
  np.testing.assert_allclose(value, jnp.linalg.norm(ab, axis=-1).mean())


def test_norm_axis_none_aggregation_type_concat_returns_concatenated_norm():
  full_tree = {
      "a": jnp.array([1, 2, 3], dtype=jnp.float32),
      "b": jnp.array([4, 5, 6], dtype=jnp.float32),
  }
  concat_norm_reduce = base.TreeReduce(
      metric=stats.Norm(
          tensor="full_tree", axis=None, aggregation_type="concat"
      )
  )
  value = concat_norm_reduce.get_state(tensor=full_tree).compute()
  assert value == pytest.approx(jnp.sqrt(sum(x**2 for x in range(1, 7))))


def test_norm_axis_none_aggregation_type_average_returns_average_norms():
  full_tree = {
      "a": jnp.array([1, 2, 3], dtype=jnp.float32),
      "b": jnp.array([4, 5, 6], dtype=jnp.float32),
  }
  concat_norm_reduce = base.TreeReduce(
      metric=stats.Norm(
          tensor="full_tree", axis=None, aggregation_type="average"
      )
  )
  value = concat_norm_reduce.get_state(tensor=full_tree).compute()
  assert value == pytest.approx(
      jnp.mean(
          jnp.asarray((
              jnp.sqrt(1**2 + 2**2 + 3**2),
              jnp.sqrt(4**2 + 5**2 + 6**2),
          ))
      )
  )


def test_norm_axis_none_aggregation_type_none_raises_warning():
  full_tree = {
      "a": jnp.array([1, 2, 3], dtype=jnp.float32),
      "b": jnp.array([4, 5, 6], dtype=jnp.float32),
  }
  concat_norm_reduce = base.TreeReduce(
      metric=stats.Norm(tensor="full_tree", axis=None, aggregation_type=None)
  )
  with pytest.warns(UserWarning):
    concat_norm_reduce.get_state(tensor=full_tree)
