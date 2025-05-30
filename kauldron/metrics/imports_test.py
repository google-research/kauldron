# Copyright 2025 The kauldron Authors.
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

"""Tests for lpips_not_found."""

from jax import numpy as jnp
from kauldron import kd
import pytest


def test_lpips_vgg_not_found():
  img_a = jnp.zeros([2, 32, 32, 3], jnp.float32)
  img_b = jnp.ones([2, 32, 32, 3], jnp.float32)
  lpips_vgg = kd.metrics.LpipsVgg(pred="pred", target="target", mask="mask")

  with pytest.raises(FileNotFoundError, match="`LpipsVgg` require"):
    lpips_vgg.get_state(pred=img_a, target=img_b, mask=None)


def test_lazy_imports():
  with pytest.raises(ImportError):
    from grand_vision.eval.metrics import clustering  # pylint: disable=g-import-not-at-top,unused-import  # pytype: disable=import-error
