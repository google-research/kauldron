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

"""LPIPS metric, see https://arxiv.org/abs/1801.03924."""
from __future__ import annotations

import dataclasses
import functools
from typing import Optional

from etils import epath
import flax
from flax import linen as nn
import flax.struct
import jax
from jax import numpy as jnp
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Float, Key, typechecked  # pylint: disable=g-multiple-import,g-importing-member


# Should be private, but changing the name of the class breaks ckpt loading...
class VggBlock(nn.Module):
  """Implementation of a block within the VGG network."""

  num_features: int
  num_layers: int

  @nn.compact
  def __call__(self, x):
    for _ in range(self.num_layers):
      x = nn.Conv(
          features=self.num_features, kernel_size=(3, 3), padding="SAME"
      )(x)
      x = jax.nn.relu(x)
    return x


# Should be private, but changing the name of the class breaks ckpt loading...
class VggNet(nn.Module):
  """Implementation of the VGG network which returns some partial results."""

  @nn.compact
  def __call__(self, x):
    assert x.shape[-2] >= 16, str(x.shape)
    assert x.shape[-3] >= 16, str(x.shape)
    outputs = []
    for num_features, num_layers in ((64, 2), (128, 2), (256, 3), (512, 3)):
      x = VggBlock(num_features=num_features, num_layers=num_layers)(x)
      outputs.append(x)
      x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    outputs.append(VggBlock(num_features=512, num_layers=3)(x))
    return tuple(outputs)


class _LpipsVgg(nn.Module):
  """Computes the LPIPS VGG score."""

  @staticmethod
  def read_params(params):
    path = epath.Path(
        "/memfile/lpips_vgg_weights_memfile/lpips_vgg_weights.bin"
    )
    return flax.serialization.from_bytes(params, path.read_bytes())

  @nn.compact
  def __call__(self, images_1, images_2, epsilon: float = 1e-5):
    """Compute the loss between inputs[0] and inputs[1].

    Both images must have height & width of at least 16 pixels.

    Args:
      images_1: A [h, w, 3] float array, assumed to range between 0 and 1.
      images_2: A [h, w, 3] float array, assumed to range between 0 and 1.
      epsilon: Used for numerical stability to avoid NaNs.

    Returns:
      A scalar containing the computed loss between images_1 & images_2.
    """
    # Note that these normalization values include a shift from from 0->1 to
    # -1->1 in the shift and scale.
    shift = (1.0 + jnp.array([-0.030, -0.088, -0.188])) / 2.0
    scale = jnp.array([0.458, 0.448, 0.450]) / 2.0
    combined = (jnp.stack((images_1, images_2)) - shift) / scale

    outputs = VggNet()(combined)
    out = 0.0
    for output in outputs:
      var = jnp.sum(
          jnp.square(output), axis=-1, keepdims=True, dtype=jnp.float32
      )
      normalization_scale = jax.lax.rsqrt(var + epsilon)
      feature_outputs = output * normalization_scale
      squared_diff = jnp.square(feature_outputs[1] - feature_outputs[0])
      res = nn.Conv(
          features=1,
          kernel_size=(1, 1),
          padding="SAME",
          use_bias=False,
      )(squared_diff[jnp.newaxis])
      # The mean over all pixels is set to float32 to avoid precision issues.
      out += jnp.mean(res, axis=(-3, -2, -1), dtype=jnp.float32)
    return out


@functools.cache
def _get_vgg_model():
  return _LpipsVgg()


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class LpipsVgg(base.Metric):
  """VGG LPIPS.

  """

  pred: Key
  target: Key
  mask: Optional[Key] = None

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred: Float["*b h w c"],
      target: Float["*b h w c"],
      mask: Optional[Float["*b 1"]] = None,
  ) -> LpipsVgg.State:
    vgg_model = _get_vgg_model()
    vgg_params = _LpipsVgg.read_params(
        vgg_model.init(
            jax.random.PRNGKey(0), jnp.ones((32, 32, 3)), jnp.ones((32, 32, 3))
        )
    )
    values = vgg_model.apply(vgg_params, pred, target)
    return self.State.from_values(values=values, mask=mask)
