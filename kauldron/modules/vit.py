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

"""Vision Transformer modules."""

from __future__ import annotations

import dataclasses
import re
from typing import Literal, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from kauldron import kontext
from kauldron.modules import attention
from kauldron.modules import input_embeddings
from kauldron.modules import knn_types
from kauldron.modules import pos_embeddings
from kauldron.modules import transformers
from kauldron.typing import Float, Initializer, Shape, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(frozen=True)
class ViTSpec:
  """Spec for the size of a Vision Transformer."""

  hidden_size: int  # Dimension of tokens passed between blocks.
  num_layers: int  # Number of trasformer blocks.
  mlp_size: int  # Hidden dimension of the MLP in each block.
  num_heads: int  # Number of attention heads.
  patch_size: Optional[int] = None  # Patch size of initial image patches.

  @classmethod
  def from_variant_string(cls, variant_str: str) -> ViTSpec:
    """Parse variant strings like "ViT-L", "B", or "Ti/16"."""
    r = re.match(
        r'^([Vv][Ii][Tt][-_])?(?P<name>[a-zA-Z]{1,2})(/(?P<patch>\d+))?$',
        variant_str,
    )
    if r is None:
      raise ValueError(f'Invalid variant string: {variant_str!r}.')
    name = r.groupdict()['name']
    spec = cls(*VIT_SIZES[name])

    patch_size = r.groupdict()['patch']
    if patch_size is not None:
      spec = dataclasses.replace(spec, patch_size=int(patch_size))
    return spec

  @property
  def kwargs(self):
    kwargs = dict(
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        mlp_size=self.mlp_size,
        num_heads=self.num_heads,
        patch_size=self.patch_size,
    )
    if self.patch_size is None:
      del kwargs['patch_size']
    return kwargs


# Reference: Table 2 of https://arxiv.org/abs/2106.04560.
VIT_SIZES = {
    'mu': (32, 1, 128, 2),
    'Ti': (192, 12, 768, 3),
    'S': (384, 12, 1536, 6),
    'M': (512, 12, 2048, 8),
    'B': (768, 12, 3072, 12),
    'L': (1024, 24, 4096, 16),
    'H': (1280, 32, 5120, 16),
    'g': (1408, 40, 6144, 16),
    'G': (1664, 48, 8192, 16),
    'e': (1792, 56, 15360, 16),
}


class VitEncoder(nn.Module):
  """Basic Vit Encoder.

  Implements:
  - extracting and linearly embedding patches
  - adding a position embedding
  - a sequence of Transformer (self-attention) layers
  - a final RMS normalization

  Attributes:
    embedding: Submodule for embedding images into a flat set of tokens.
    pos_embedding: Position Embeddings to add to the embedded tokens.
    layers: Sequence of transformer blocks to apply.
    encoder_norm: Normalization to be applied at the end of the encoder.
    prepend_cls_token: Whether to prepend a cls token after the position
      embeddings.
    cls_token_init: initializer for the cls token (if present).
  """

  # Submodules.
  layers: Sequence[knn_types.TransformerBlock]
  embedding: knn_types.ImageTokenizer
  pos_embedding: knn_types.PositionEmbedding = pos_embeddings.LearnedEmbedding()
  encoder_norm: Optional[knn_types.NormModule] = nn.RMSNorm()

  # Parameters.
  prepend_cls_token: bool = False
  cls_token_init: Initializer = nn.initializers.zeros

  # Keys. If top-level module this will be used to gather args for __call__.
  image: kontext.Key = kontext.REQUIRED  # e.g. 'batch.image'

  @typechecked
  @nn.compact
  def __call__(self, image: Float['*b h w c']) -> Float['*b n d']:
    # Embed the inputs into tokens of shape `*b n1 d`. Note that it's n1 instead
    # of n here, because there might be an additional cls token.
    tokens = self.embedding(image)
    check_type(tokens, Float['*b n1 d'])

    # Add position embeddings.
    tokens = tokens + self.pos_embedding(tokens.shape, axis=-2)

    # Add a cls token (optional).
    if self.prepend_cls_token:
      cls_tok = self.param('cls', self.cls_token_init, Shape('d'), tokens.dtype)
      cls_tok = jnp.broadcast_to(cls_tok, Shape('*b 1 d'))
      tokens = jnp.concatenate([cls_tok, tokens], axis=-2)

    check_type(tokens, Float['*b n d'])

    # Self Attention Layers.
    for layer in self.layers:
      tokens = layer(tokens)

    # Add a final normalization (optional).
    if self.encoder_norm:
      tokens = self.encoder_norm(tokens)

    return tokens

  @classmethod
  def from_variant_str(cls, variant_str: str, **kwargs) -> VitEncoder:
    vit_spec = ViTSpec.from_variant_string(variant_str)
    return cls.from_spec(**(vit_spec.kwargs | kwargs))

  @classmethod
  def from_spec(
      cls,
      num_heads: int,
      hidden_size: int,
      num_layers: int,
      patch_size: int | tuple[int, int],
      mlp_size: Optional[int] = None,
      block_type=transformers.PreNormBlock,
      dtype=jnp.float32,
      **kwargs,
  ):
    if isinstance(patch_size, int):
      patch_size = (patch_size, patch_size)

    return cls(
        embedding=input_embeddings.PatchifyEmbed(
            patch_size=patch_size, hidden_size=hidden_size
        ),
        layers=tuple(
            block_type(  # pylint: disable=g-complex-comprehension
                attention_norm=nn.LayerNorm(dtype=dtype),
                mlp_norm=nn.LayerNorm(dtype=dtype),
                attention=attention.ImprovedMultiHeadDotProductAttention(
                    num_heads=num_heads,
                ),
                mlp=transformers.TransformerMLP(hidden_size=mlp_size),
            )
            for _ in range(num_layers)
        ),
        **kwargs,
    )


class Vit(nn.Module):
  """Basic Vision Transformer classifer with GAP."""

  # Submodules.
  encoder: nn.Module

  # Parameters.
  num_classes: int = 1000
  init_head_bias: Initializer = nn.initializers.zeros
  mode: Literal['gap', 'cls_token', 'cls_token_forced'] = 'gap'

  # Keys. If top-level module this will be used to gather args for __call__.
  image: kontext.Key = kontext.REQUIRED  # E.g. 'batch.image'.

  @typechecked
  @nn.compact
  def __call__(
      self,
      image: Float['*b h w c'],
  ) -> dict[str, Float['*b num_classes']]:
    tokens = self.encoder(image)
    check_type(tokens, Float['*b n feat'])

    # Determine the input to the classification layer.
    if self.mode == 'gap':
      encoder_out = jnp.mean(tokens, axis=-2, keepdims=False)
    elif self.mode == 'cls_token':
      encoder_out = tokens[..., 0, :]
      if not self.encoder.prepend_cls_token:
        raise ValueError(
            "For model 'cls_token', the encoder should have prepend_cls_token"
            " set to True (or use 'cls_token_forced' instead)."
        )
    elif self.mode == 'cls_token_forced':
      encoder_out = tokens[..., 0, :]  # same as above but without check
    else:
      raise ValueError(
          f'Unknown {self.mode=}. '
          'Has to be one of ["gap", "cls_token", "cls_token_forced"].'
      )
    check_type(encoder_out, Float['*b feat'] | Float['*b n 2*feat'])

    # Add a dense hidden layer with tanh activations.
    pre_logits = nn.Dense(
        features=encoder_out.shape[-1],
        name='pre_logits',
    )(encoder_out)
    pre_logits = nn.tanh(pre_logits)
    logits = nn.Dense(
        features=self.num_classes,
        name='classifier_head',
        bias_init=self.init_head_bias,
    )(pre_logits)
    return {'logits': logits}
