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

"""Deprecated Modules."""

from __future__ import annotations

import re
from typing import Optional
import warnings

import einops
from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron.modules import knn_types
from kauldron.modules import pos_embeddings
from kauldron.typing import Axes, Bool, Float, Initializer, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import interms_property


class VitEncoder(nn.Module):
  """Deprecated Vit Encoder.

  Use kd.modules.vit.VitEncoder instead

  Implements:
  - extracting and linearly embedding patches
  - adding a position embedding
  - a sequence of Transformer (self-attention) layers
  - a final RMS normalization
  """

  patch_size: tuple[int, int]
  hidden_size: int
  num_layers: int
  block_template: nn.Module

  pos_embedding: knn_types.PositionEmbedding = pos_embeddings.LearnedEmbedding()

  # If used as top-module this key will be used to fill the argument of __call__
  images: kontext.Key = 'batch.image'

  @typechecked
  @nn.compact
  def __call__(self, images: Float['*b h w c']) -> Float['*b n d']:
    warnings.warn(
        'Deprecated! Use kd.modules.vit.VitEncoder instead.',
        category=DeprecationWarning,
    )
    # Linear patch embedding with a Conv layer.
    tokens = nn.Conv(
        name='patchifier',
        features=self.hidden_size,
        kernel_size=self.patch_size,
        strides=self.patch_size,
        padding='VALID',
    )(images)
    tokens = einops.rearrange(tokens, '... ht wt feat -> ... (ht wt) feat')
    tokens = tokens + self.pos_embedding(tokens.shape, axis=-2)

    # Self Attention Layers
    for i in range(self.num_layers):
      tokens = self.block_template.clone(parent=self, name=f'layer_{i}')(tokens)

    tokens = nn.RMSNorm(dtype=tokens.dtype, name='norm_encoder')(tokens)

    return tokens

  @classmethod
  def from_variant_str(cls, variant: str, **kwargs) -> VitEncoder:
    warnings.warn(
        'Deprecated! Use kd.modules.vit.VitEncoder instead.',
        category=DeprecationWarning,
    )
    r = re.match(
        r'^([Vv]i[Tt]-)?(?P<name>[a-zA-Z]{1,2})(/(?P<patch>\d+))?$', variant
    )
    if r is None:
      raise ValueError(f'Invalid variant string: {variant!r}.')
    patch_size = int(r.groupdict()['patch'] or 16)
    name = r.groupdict()['name']
    # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
    hidden_size, num_layers, mlp_size, num_heads = {
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
    }[name]
    block_kwargs = {
        'mlp_size': mlp_size,
        'num_heads': num_heads,
        'qk_size': hidden_size,
    }
    block_kwargs.update(kwargs)
    return cls(
        patch_size=(patch_size, patch_size),
        hidden_size=hidden_size,
        num_layers=num_layers,
        block_template=ParallelSelfAttentionBlock(**block_kwargs),
    )


class Vit(nn.Module):
  """Deprecated Vision Transformer classifer with GAP.

  Use kd.modules.vit.Vit instead.
  """

  encoder: nn.Module
  num_classes: int = 1000
  image: kontext.Key = 'batch.image'
  init_head_bias: Initializer = nn.initializers.zeros

  @typechecked
  @nn.compact
  def __call__(
      self, image: Float['*b h w c']
  ) -> dict[str, Float['*b num_classes']]:
    warnings.warn(
        'Deprecated! Use kd.modules.vit.Vit instead.',
        category=DeprecationWarning,
    )
    tokens = self.encoder(image)
    check_type(tokens, Float['*b n feat'])
    gap = jnp.mean(tokens, axis=-2)
    # add a dense hidden layer with tanh activations following:
    pre_logits = nn.Dense(
        features=tokens.shape[-1],
        dtype=gap.dtype,
        name='pre_logits',
    )(gap)
    pre_logits = nn.tanh(pre_logits)
    logits = nn.Dense(
        features=self.num_classes,
        dtype=pre_logits.dtype,
        name='classifier_head',
        bias_init=self.init_head_bias,
    )(pre_logits)
    return {'logits': logits}


class ParallelSelfAttentionBlock(nn.Module):
  """Parallel self attention (see Vit22B paper).

  """

  mlp_size: int
  num_heads: int
  qk_size: int
  softmax_axis: Axes = -1

  @typechecked
  @nn.compact
  def __call__(self, tokens: Float['*b n d']) -> Float['*b n d']:
    warnings.warn(
        'Deprecated! Use kd.modules.transformers.ParallelAttentionBlock'
        ' instead.',
        category=DeprecationWarning,
    )
    width = tokens.shape[-1]
    norm_tokens = nn.RMSNorm(dtype=tokens.dtype, name='norm_in')(tokens)
    post_att = ImprovedMHDPAttention(
        num_heads=self.num_heads,
        qk_size=self.qk_size,
        name='MHDPA',
        softmax_axis=self.softmax_axis,
    )(inputs_q=norm_tokens, inputs_kv=norm_tokens)
    h = nn.gelu(
        nn.Dense(self.mlp_size, dtype=norm_tokens.dtype, name='MLP_in')(
            norm_tokens
        )
    )
    out = nn.Dense(width, dtype=h.dtype, name='MLP_out')(h)
    return tokens + out + post_att


class ImprovedMHDPAttention(nn.Module):
  """Multi-head dot-product attention.

  Simplified nn.MultiheadDotProductAttention with a few modifications:
    - include a softmax axis
    - include normalization of keys and queries
    - dropped out support for dropout

  Attributes:
    num_heads: Number of attention heads.
    qk_size: Total dimension of the keys and queries.
    v_size: Total dimension of the values. Defaults to qk_size.
    softmax_axis: The axis over which the softmax is taken. defaults to -1 which
      is the keys axis. For Slot-Attention set to -2 (queries).
  """

  num_heads: int
  qk_size: int
  v_size: Optional[int] = None
  softmax_axis: Axes = -1

  interms = interms_property.interms_property()

  @typechecked
  @nn.compact
  def __call__(
      self,
      inputs_q: Float['*b q d1'],
      inputs_kv: Float['*b k d2'],
      bias: Optional[Float['*b #h #q #k']] = None,
      mask: Optional[Bool['*b #h #q #k']] = None,
  ) -> Float['*b q d1']:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: Input tokens from which queries are computed.
      inputs_kv: Input tokens from which the keys and queries are computed.
      bias: Bias for the attention weights. This can be used for incorporating
        causal masks, padding masks, proximity bias, etc.
      mask: Attention mask, where attention weights are masked out if their mask
        value is `False`.

    Returns:
      output tokens (linear projection of an attention weighted average of value
      tokens per query).
    """
    warnings.warn(
        'Deprecated! Use kd.modules.attention.ImprovedMHDPAttention instead.',
        category=DeprecationWarning,
    )
    v_size = self.qk_size if self.v_size is None else self.v_size

    if self.qk_size % self.num_heads:
      raise ValueError(f'{self.num_heads=} must divide {self.qk_size=}.')
    if v_size % self.num_heads:
      raise ValueError(f'{v_size=} must divide {self.num_heads=}.')

    # Project inputs_q to multi-headed queries and keys.
    # dimensions are then [B..., Q, H, qk_size]
    query = nn.DenseGeneral(
        features=(self.num_heads, self.qk_size // self.num_heads),
        use_bias=False,
        dtype=inputs_q.dtype,
        name='dense_query',
    )(inputs_q)
    key = nn.DenseGeneral(
        features=(self.num_heads, self.qk_size // self.num_heads),
        use_bias=False,
        dtype=inputs_kv.dtype,
        name='dense_key',
    )(inputs_kv)

    # Normalize keys and queries before attention.
    # see Gilmer et al. 2023
    # Intriguing Properties of Transformer Training Instabilities
    query = nn.RMSNorm(dtype=query.dtype, name='norm_query')(query)
    key = nn.RMSNorm(dtype=query.dtype, name='norm_key')(key)

    value = nn.DenseGeneral(
        features=(self.num_heads, v_size // self.num_heads),
        use_bias=False,
        dtype=inputs_kv.dtype,
        name='dense_value',
    )(inputs_kv)

    # Compute attention weights.
    attn_weights = dot_product_attention_weights(
        query, key, self.softmax_axis, bias=bias, mask=mask
    )
    # accessible as `interms.[path.to.this.module].attention[0]`
    self.interms['attention'] = attn_weights

    # Return weighted sum over values for each query position.
    x = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

    # Back to the original input dimensions.
    out = nn.DenseGeneral(
        features=inputs_q.shape[-1],
        axis=(-2, -1),
        use_bias=True,
        dtype=x.dtype,
        name='dense_out',
    )(x)

    return out


@typechecked
def dot_product_attention_weights(
    query: Float['*b q h d'],
    key: Float['*b k h d'],
    softmax_axes: Axes = -1,
    bias: Optional[Float['*b #h #q #k']] = None,
    mask: Optional[Bool['*b #h #q #k']] = None,
) -> Float['*b h q k']:
  """Computes dot-product attention weights given query and key.

  q: number of queries, k: number of keys, h: number of heads
  d: dimension of keys/queries

  Args:
    query: Queries for calculating attention
    key: Keys for calculating attention.
    softmax_axes: The axes over which the softmax is taken. defaults to -1 which
      is the keys axis. For Slot-Attention set to -2 (queries).
    bias: Bias for the attention weights. This should be broadcastable to the
      shape `[*b h q k]`. This can be used for incorporating causal masks,
      padding masks, proximity bias, etc.
    mask: Mask for the attention weights. This should be broadcastable to the
      shape `[*b h q k]`. This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value is
      `False`.

  Returns:
    Attention weights of shape `[*b h q k]`.
  """
  query = query / jnp.sqrt(query.shape[-1])
  attn_weights = jnp.einsum('...qhd,...khd->...hqk', query, key)

  if bias is not None:
    attn_weights = attn_weights + bias

  if mask is not None:
    big_neg = jnp.finfo(query.dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

  attn_weights = jax.nn.softmax(attn_weights, axis=softmax_axes)

  return attn_weights
