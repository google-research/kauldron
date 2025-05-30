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

"""Nerf module."""

from __future__ import annotations

import dataclasses

from flax import linen as nn
from kauldron import kontext
from projects.nerf import math
from projects.nerf.core import structs
from projects.nerf.core.typing import ActivationFn  # pylint: disable=g-importing-member
from projects.nerf.modules import encoders
from kauldron.typing import Float  # pylint: disable=g-multiple-import
import visu3d as v3d


class NerfRender(nn.Module):  # pytype: disable=invalid-function-definition
  """."""
  _: dataclasses.KW_ONLY

  ray: kontext.Key = kontext.REQUIRED

  point_net: PointNetwork

  @nn.compact
  def __call__(self, ray: v3d.Ray) -> structs.RayPreds:
    # TODO(epot): How to chunk the rays to be given to the model ? Can this
    # be done inside `jax.jit` ?
    sample_rng = self.make_rng('samples')

    sample_depths, sample_pos = math.sample_along_rays(
        ray_origins=ray.pos,
        ray_directions=ray.dir,
        # TODO(epot): This depend on the dataset
        near=0.1,
        far=4.0,
        sample_count=64,  # TODO(epot): Customizable
        rng=sample_rng,
    )

    point_pred = self.point_net(sample_pos)

    vol_res = math.volume_rendering(
        sample_values=point_pred.rgb,
        sample_density=point_pred.density.squeeze(axis=-1),
        depths=sample_depths,
        # TODO(epot): White background ?
    )
    return structs.RayPreds(
        rgb=vol_res.ray_values,
        depth=vol_res.ray_depth[..., None],
    )


class MultiLevelRender(nn.Module):
  """Coarse and fine network."""

  pass


class PointNetwork(nn.Module):  # pytype: disable=invalid-function-definition
  """Network that predict a single point in space."""
  _: dataclasses.KW_ONLY

  width: int = 256
  num_layers: int = 8

  body: nn.Module
  head: nn.Module = dataclasses.field(default_factory=lambda: PointHead)  # pytype: disable=name-error

  @nn.compact
  def __call__(self, sample_pos: Float['*bs 3']) -> structs.PointPreds:  # [*bs]
    # TODO(epot): Customizable `num_basis`
    sample_enc = encoders.fourier_enc(sample_pos, num_basis=10)

    # TODO(epot): Original jaxnerf implementation re-concatenate the original
    # position, but shouldn't the pos encoding be enough ?
    # sample_enc = jnp.concatenate([sample_pos, sample_enc], axis=-1)

    x = sample_enc
    x = self.body(x)
    x = self.head(x)
    return x


class PointHead(nn.Module):
  rgb_activation: ActivationFn = nn.sigmoid
  density_activation: ActivationFn = nn.relu

  @nn.compact
  def __call__(self, x: Float['*bs 3']) -> structs.PointPreds:  # [*bs]
    rgb = nn.Dense(3)(x)
    rgb = self.rgb_activation(rgb)

    density = nn.Dense(1)(x)
    density = self.density_activation(density)
    return structs.PointPreds(
        rgb=rgb,
        density=density,
    )
