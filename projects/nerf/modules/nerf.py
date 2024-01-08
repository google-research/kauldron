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

"""Nerf module."""

import dataclasses

from flax import linen as nn
from kauldron import kontext
from projects.nerf.core import structs
import visu3d as v3d


class NerfRender(nn.Module):  # pytype: disable=invalid-function-definition
  """."""
  _: dataclasses.KW_ONLY

  ray: kontext.Key = kontext.REQUIRED

  mlp: nn.Module

  @nn.compact
  def __call__(self, ray: v3d.Ray) -> structs.RayPreds:
    # TODO(epot): How to chunk the rays to be given to the model ? Can this
    # be done inside `jax.jit` ?
    raise NotImplementedError()


class PointRender(nn.Module):
  """."""

  @nn.compact
  def __call__(self, point: v3d.Point3d) -> structs.PointPreds:
    raise NotImplementedError()
