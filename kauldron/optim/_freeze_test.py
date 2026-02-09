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

import jax.numpy as jnp
from kauldron import kd
import optax


def test_partial_updates():
  optimizer = kd.optim.partial_updates(
      optax.adam(learning_rate=1e-3),
      mask=kd.optim.select('lora'),
  )

  params = {
      'a': {
          'lora': {
              'x': jnp.zeros((2,)),
              'y': jnp.zeros((2,)),
          }
      },
      'x': jnp.zeros((2,)),
      'y': jnp.zeros((2,)),
  }

  assert kd.optim._freeze._make_labels(params, kd.optim.select('lora')) == {
      'a': {
          'lora': {
              'x': 'train',
              'y': 'train',
          }
      },
      'x': 'freeze',
      'y': 'freeze',
  }

  # TODO(epot): Could check the state params is empty for frozen params.
  optimizer.init(params)
