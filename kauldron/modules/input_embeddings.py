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

"""Modules for embedding the inputs."""

from __future__ import annotations

import einops
from flax import linen as nn
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class Patchify(nn.Module):
  """Patchify an image, as in ViT (without linear embedding)."""

  patch_size: tuple[int, int]

  @typechecked
  @nn.compact
  def __call__(self, images: Float['*b h w c']) -> Float['*b n d']:
    tokens = einops.rearrange(
        images,
        '... (ht hp) (wt wp) C -> ... (ht wt) (hp wp C)',
        hp=self.patch_size[0],
        wp=self.patch_size[1],
    )
    return tokens


class PatchifyEmbed(nn.Module):
  """Patchify and linearly embed and image, as in ViT."""

  patch_size: tuple[int, int]
  hidden_size: int

  @typechecked
  @nn.compact
  def __call__(self, images: Float['*b h w c']) -> Float['*b n d']:
    # Linear patch embedding with a Conv layer.
    tokens = nn.Conv(
        name='patchifier',
        features=self.hidden_size,
        kernel_size=self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        dtype=images.dtype
    )(images)
    tokens = einops.rearrange(tokens, '... ht wt feat -> ... (ht wt) feat')
    return tokens
