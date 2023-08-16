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
import dataclasses

import jax.numpy as jnp
from kauldron.losses import base
from kauldron.typing import Key  # pylint: disable=g-importing-member
from kauldron.utils import core
import numpy as np


# --------- Test a custom base Loss -------
@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class TenX(base.Loss):
  x: Key = 'batch.x'

  def get_values(self, x):
    return 10 * x


def test_ten_x_loss():
  l = TenX()
  x = np.ones((2, 3))
  y = l(x=x)
  np.testing.assert_allclose(y, 10)


def test_ten_x_loss_weight():
  l = TenX(weight=2.5)  # pylint: disable=unexpected-keyword-arg
  x = jnp.ones((2, 3))
  y = l(x=x)
  np.testing.assert_allclose(y, 10 * 2.5)


def test_ten_x_loss_apply_to_context():
  l = TenX()
  x = jnp.ones((2, 3))
  context = core.Context(step=100, batch={'x': x})
  y = l(context=context)
  np.testing.assert_allclose(y, 10)


def test_ten_x_loss_weight_schedule():
  l = TenX(weight=lambda step: step / 20)  # pylint: disable=unexpected-keyword-arg
  x = jnp.ones((2, 3))
  context = core.Context(step=100, batch={'x': x})
  y = l(context=context)
  np.testing.assert_allclose(y, 10 * 5)


def test_ten_x_masked_loss():
  l = TenX()
  l_weight = TenX(weight=3.0)
  x = np.ones((2, 3))
  x[1, :] = 2
  mask = np.ones_like(x)
  mask[1, :] = 0
  mask = jnp.array(mask)

  np.testing.assert_allclose(l(x=x), 15.0)
  np.testing.assert_allclose(l(x=x, mask=mask), 10.0)
  np.testing.assert_allclose(l_weight(x=x), 45.0)
  np.testing.assert_allclose(l_weight(x=x, mask=mask), 30.0)
