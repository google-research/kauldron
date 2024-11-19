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

from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron import kd
import numpy as np


def test_external():
  model = kd.nn.ExternalModule(
      model=nn.Dropout(0.5),
      keys={
          'inputs': 'a',
      },
      train_kwarg_name='~deterministic',
  )

  inputs = jnp.ones((5, 5))
  input_kwargs = kd.kontext.resolve_from_keyed_obj(
      {'a': inputs, 'b': jnp.zeros(())}, model
  )
  out_train = model.apply(
      {},
      **input_kwargs,
      is_training_property=True,
      rngs={'dropout': jax.random.PRNGKey(0)},
  )
  out_eval = model.apply(
      {},
      **input_kwargs,
      is_training_property=False,
  )

  assert not np.array_equal(out_train, inputs)
  np.testing.assert_array_equal(out_eval, inputs)
