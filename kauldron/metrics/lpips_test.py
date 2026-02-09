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

"""Tests for LPIPS metric."""

from jax import numpy as jnp
from kauldron.metrics import lpips


def test_lpips_vgg():
  img_a = jnp.zeros([2, 32, 32, 3], jnp.float32)
  img_b = jnp.ones([2, 32, 32, 3], jnp.float32)
  lpips_vgg = lpips.LpipsVgg(pred="pred", target="target", mask="mask")
  state_1 = lpips_vgg.get_state(pred=img_a, target=img_b, mask=None)
  unused_result_1 = state_1.compute()

  # Test for images without batch dimension.
  state_2 = lpips_vgg.get_state(pred=img_a[0], target=img_b[0], mask=None)
  unused_result_2 = state_2.compute()
