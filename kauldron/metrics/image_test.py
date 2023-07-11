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

"""Tests for metrics/image."""

from jax import numpy as jnp
from kauldron.metrics import image


def test_lpips_vgg():
  img_a = jnp.zeros([2, 32, 32, 3], jnp.float32)
  img_b = jnp.ones([2, 32, 32, 3], jnp.float32)
  lpips_vgg = image.LpipsVgg(pred="pred", target="target", mask="mask")
  state = lpips_vgg.get_state(pred=img_a, target=img_b, mask=None)
  unused_result = lpips_vgg.compute(state)
