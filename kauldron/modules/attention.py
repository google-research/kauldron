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

"""Attention layers."""

from __future__ import annotations

from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import kauldron.modules as knn
from kauldron.typing import Axes, Bool, DType, Float, Initializer, typechecked  # pylint: disable=g-multiple-import,g-importing-member


def softmax(
    x: Float['*a'], axis: Axes = -1, dtype: DType | None = jnp.float32
) -> Float['*a']:
  if dtype is None:
    dtype = x.dtype
  return jax.nn.softmax(x.astype(dtype), axis=axis).astype(x.dtype)


class MultiHeadDotProductAttention(nn.MultiHeadDotProductAttention):
  """Wrapper around `nn.MultiHeadDotProductAttention` using `knn.train_property`."""

  is_training = knn.train_property()

  def __post_init__(self):
    super().__post_init__()
    if self.deterministic is not None:
      raise ValueError(
          '`kd.nn.Dropout` should not use `deterministic`. Instead the '
          'training mode is set through `is_training_property`. See '
          '`kd.nn.train_property`.'
      )

  @typechecked
  @nn.compact
  def __call__(
      self,
      inputs_q: Float['*b q dq'],
      inputs_k: Optional[Float['*b k dk']] = None,
      inputs_v: Optional[Float['*b k dv']] = None,
      *,
      mask: Optional[Bool['*b #heads #q #k']] = None,
  ) -> Float['*b q do']:
    return super().__call__(
        inputs_q=inputs_q,
        inputs_k=inputs_k,
        inputs_v=inputs_v,
        deterministic=not self.is_training,
        mask=mask,
    )


@typechecked
def dot_product_attention_weights(
    query: Float['*b q h d'],
    key: Float['*b k h d'],
    softmax_axis: Axes = -1,
    bias: Optional[Float['*b #h #q #k']] = None,
    mask: Optional[Bool['*b #h #q #k']] = None,
    softmax_dtype: DType | None = jnp.float32,
) -> Float['*b h q k']:
  """Computes dot-product attention weights given query and key.

  q: number of queries, k: number of keys, h: number of heads
  d: dimension of keys/queries

  Args:
    query: Queries for calculating attention
    key: Keys for calculating attention.
    softmax_axis: The axes over which the softmax is taken. defaults to -1 which
      is the keys axis. For Slot-Attention set to -2 (queries).
    bias: Bias for the attention weights. This should be broadcastable to the
      shape `[*b h q k]`. This can be used for incorporating causal masks,
      padding masks, proximity bias, etc.
    mask: Mask for the attention weights. This should be broadcastable to the
      shape `[*b h q k]`. This can be used for incorporating causal masks.
      Attention weights are masked out if their corresponding mask value is
      `False`.
    softmax_dtype: The dtype for the softmax operation. If None, the dtype of
      the input is used.

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

  attn_weights = softmax(attn_weights, axis=softmax_axis, dtype=softmax_dtype)

  return attn_weights


class ImprovedMultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

  Simplified nn.MultiheadDotProductAttention with a few modifications:
    - include a softmax axis
    - accept an (additive) bias for the attention weights (in addition to mask)
    - dropped support for dropout
    - add attention weights to interms as "interms.PATH.TO.LAYER.attn_weights"


  Attributes:
    num_heads: Number of attention heads.
    qk_size: Total dimension of the keys and queries.
    v_size: Total dimension of the values. Defaults to qk_size.
    softmax_axis: The axis over which the softmax is taken. defaults to -1 which
      is the keys axis. For Slot-Attention set to -2 (queries).
  """

  num_heads: int
  qk_features: Optional[int] = None
  v_features: Optional[int] = None
  out_features: Optional[int] = None
  softmax_axis: Axes = -1

  normalize_qk: bool = False

  kernel_init: Initializer = nn.initializers.lecun_normal()
  bias_init: Initializer = nn.initializers.zeros_init()
  use_bias: bool = True
  attn_weights_fn: Callable[..., Float['...']] = dot_product_attention_weights
  decode: bool = False

  interms = knn.interms_property()

  @typechecked
  @nn.compact
  def __call__(
      self,
      inputs_q: Float['*b q dq'],
      inputs_k: Optional[Float['*b kv dk']] = None,  # defaults to inputs_q
      inputs_v: Optional[Float['*b kv dv']] = None,  # defaults to inputs_k
      *,
      bias: Optional[Float['*b #num_heads #q #kv']] = None,
      mask: Optional[Bool['*b #num_heads #q #kv']] = None,
  ) -> Float['*b q do']:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: Input tokens from which queries are computed.
      inputs_k: Input tokens from which the keys are computed (defaults to
        inputs_q).
      inputs_v: Input tokens from which the values are computed (defaults to
        inputs_k).
      bias: Bias for the attention weights. This can be used for incorporating
        causal masks, padding masks, proximity bias, etc.
      mask: Attention mask, where attention weights are masked out if their mask
        value is `False`.

    Returns:
      output tokens (linear projection of an attention weighted average of value
      tokens per query).
    """
    qk_features = self.qk_features or inputs_q.shape[-1]
    v_features = self.v_features or qk_features

    if qk_features % self.num_heads:
      raise ValueError(f'{self.num_heads=} must divide {qk_features=}.')
    if v_features % self.num_heads:
      raise ValueError(f'{self.num_heads=} must divide {v_features=}.')

    if inputs_k is None:
      if inputs_v is not None:
        raise ValueError('inputs_k cannot be None if inputs_v is given.')
      inputs_k = inputs_q
    if inputs_v is None:
      inputs_v = inputs_k

    # Project inputs_q to multi-headed queries and keys.
    # dimensions are then [*b q h qk_size]
    def dense(name, x, features):
      return nn.DenseGeneral(
          features=(self.num_heads, features // self.num_heads),
          kernel_init=self.kernel_init,
          bias_init=self.bias_init,
          use_bias=self.use_bias,
          dtype=x.dtype,
          name=name,
      )(x)

    query = dense('query', inputs_q, qk_features)
    key = dense('key', inputs_k, qk_features)
    value = dense('value', inputs_v, v_features)

    if self.normalize_qk:
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = nn.LayerNorm(
          name='query_norm', use_bias=False, dtype=query.dtype)(query)
      key = nn.LayerNorm(
          name='key_norm', use_bias=False, dtype=key.dtype)(key)

    # Compute attention weights.
    attn_weights = self.attn_weights_fn(  # pylint: disable=redundant-keyword-arg
        query=query,
        key=key,
        softmax_axis=self.softmax_axis,
        bias=bias,
        mask=mask,
    )

    # accessible as `interms.[path.to.this.module].attn_weights[0]`
    self.interms['attn_weights'] = attn_weights

    # Return weighted sum over values for each query position.
    x = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)

    # Back to the original input dimensions.
    return nn.DenseGeneral(
        features=self.out_features or inputs_q.shape[-1],
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        use_bias=self.use_bias,
        dtype=x.dtype,
        name='out',
    )(x)
