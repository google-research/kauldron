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

"""Encoder classes."""

import einops
from jax import numpy as jnp
from kauldron.typing import Float  # pylint: disable=g-multiple-import,g-importing-member


# Currently, assume a
# ould try a per-axis embedding: `axis={-1: FourierInfo(degree=10, ...)}`
def fourier_enc(
    coords: Float['*b coord'],
    *,
    num_basis: int,
) -> Float['*b 2*coord*num_basis']:
  """Compute the fourier enconding.

  Args:
    coords: Positions to encode
    num_basis: Number of fourier features

  Returns:
    The fourier embeddings
  """

  scales = jnp.asarray([2**i for i in range(num_basis)])

  # (... coords) @ (basis) -> (... coords basis)
  x = jnp.einsum('...,b->...b', coords, scales)

  x = einops.rearrange(x, '... coord basis -> ... (coord basis)')

  # Obtain Fourier features as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
  # (vectorize `cos` into a single `sin` call)
  # Instead of interleave sin and cos
  # (`sin(x0), cos(x0), sin(x1), cos(x1),...`), they are concatenated
  # (`sin(x0), sin(x1), ..., cos(x0), cos(x1), ...`)
  return jnp.sin(jnp.concatenate([x, x + 0.5 * jnp.pi], axis=-1))
