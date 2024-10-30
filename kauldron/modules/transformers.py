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

"""Transformer Modules."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import flax.linen as nn
from kauldron.modules import knn_types
import kauldron.modules.attention as knn_attn
from kauldron.typing import Bool, Float, Initializer, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class TransformerMLP(nn.Module):
  """Simple MLP with a single hidden layer for use in Transformer blocks."""

  hidden_size: Optional[int] = None  # Defaults to 4 times input dims.
  activation_fn: knn_types.ActivationFunction = nn.gelu
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: Float['*b d']) -> Float['*b d']:
    d = inputs.shape[-1]
    hidden_size = 4 * d if self.hidden_size is None else self.hidden_size
    h = nn.Dense(
        features=hidden_size,
        name='dense_in',
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=inputs.dtype,
    )(inputs)
    h = self.activation_fn(h)
    return nn.Dense(
        features=inputs.shape[-1],
        name='dense_out',
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        dtype=h.dtype,
    )(h)


class PreNormBlock(nn.Module):
  """Pre-LN Transformer layer (default transformer layer)."""

  attention: knn_types.AttentionModule
  mlp: nn.Module = dataclasses.field(default_factory=TransformerMLP)

  attention_norm: knn_types.NormModule = dataclasses.field(
      default_factory=nn.LayerNorm
  )
  mlp_norm: knn_types.NormModule = dataclasses.field(
      default_factory=nn.LayerNorm
  )

  @typechecked
  @nn.compact
  def __call__(
      self,
      tokens: Float['*b n d'],
      attention_mask: Bool['*b 1 n n'] | None = None,
  ) -> Float['*b n d']:
    norm_tokens = self.attention_norm(tokens)  # pylint: disable=not-callable
    tokens += self.attention(
        inputs_q=norm_tokens,
        inputs_k=norm_tokens,
        inputs_v=norm_tokens,
        mask=attention_mask,
    )
    norm_tokens = self.mlp_norm(tokens)  # pylint: disable=not-callable
    return tokens + self.mlp(norm_tokens)  # pylint: disable=not-callable

  @classmethod
  def from_spec(
      cls,
      num_heads: int,
      mlp_size: Optional[int] = None,
      normalize_qk: bool = True,
      attn_kwargs: Optional[dict[str, Any]] = None,
      mlp_kwargs: Optional[dict[str, Any]] = None,
      **kwargs,
  ) -> PreNormBlock:
    attn_kwargs = attn_kwargs or {}
    mlp_kwargs = mlp_kwargs or {}
    return cls(
        attention=knn_attn.ImprovedMultiHeadDotProductAttention(
            num_heads=num_heads,
            normalize_qk=normalize_qk,
            **attn_kwargs,
        ),
        mlp=TransformerMLP(
            hidden_size=mlp_size,
            **mlp_kwargs,
        ),
        **kwargs,
    )


class PostNormBlock(nn.Module):
  """Post-LN Transformer layer (not recommended)."""

  attention: knn_types.AttentionModule
  mlp: nn.Module = dataclasses.field(default_factory=TransformerMLP)

  attention_norm: knn_types.NormModule = dataclasses.field(
      default_factory=nn.LayerNorm
  )
  mlp_norm: knn_types.NormModule = dataclasses.field(
      default_factory=nn.LayerNorm
  )

  @typechecked
  @nn.compact
  def __call__(
      self,
      tokens: Float['*b n d'],
      attention_mask: Bool['*b 1 n n'] | None = None,
  ) -> Float['*b n d']:
    tokens += self.attention(
        inputs_q=tokens, inputs_k=tokens, inputs_v=tokens, mask=attention_mask
    )
    norm_tokens = self.attention_norm(tokens)  # pylint: disable=not-callable
    tokens += self.mlp(norm_tokens)  # pylint: disable=not-callable
    return self.mlp_norm(tokens)  # pylint: disable=not-callable

  @classmethod
  def from_spec(
      cls,
      num_heads: int,
      mlp_size: Optional[int] = None,
      normalize_qk: bool = True,
      attn_kwargs: Optional[dict[str, Any]] = None,
      mlp_kwargs: Optional[dict[str, Any]] = None,
      **kwargs,
  ) -> PostNormBlock:
    attn_kwargs = attn_kwargs or {}
    mlp_kwargs = mlp_kwargs or {}
    return cls(
        attention=knn_attn.ImprovedMultiHeadDotProductAttention(
            num_heads=num_heads,
            normalize_qk=normalize_qk,
            **attn_kwargs,
        ),
        mlp=TransformerMLP(
            hidden_size=mlp_size,
            **mlp_kwargs,
        ),
        **kwargs,
    )


class ParallelAttentionBlock(nn.Module):
  """Parallel self attention (see Vit22B paper: arxiv.org/abs/2302.05442)."""

  attention: knn_types.AttentionModule
  mlp: nn.Module = dataclasses.field(default_factory=TransformerMLP)

  # Careful: The default values here are overwritten in `VitEncoder.from_spec`
  attention_norm: knn_types.NormModule = dataclasses.field(
      default_factory=nn.RMSNorm
  )
  mlp_norm: knn_types.NormModule = dataclasses.field(
      default_factory=nn.LayerNorm
  )

  @typechecked
  @nn.compact
  def __call__(self, tokens: Float['*b n d']) -> Float['*b n d']:
    norm_tokens = self.attention_norm(tokens)
    post_att = self.attention(
        inputs_q=norm_tokens, inputs_k=norm_tokens, inputs_v=norm_tokens
    )
    norm_tokens = self.mlp_norm(tokens)
    post_mlp = self.mlp(norm_tokens)  # pylint: disable=not-callable
    return tokens + post_att + post_mlp

  @classmethod
  def from_spec(
      cls,
      num_heads: int,
      mlp_size: Optional[int] = None,
      normalize_qk: bool = True,
      attn_kwargs: Optional[dict[str, Any]] = None,
      mlp_kwargs: Optional[dict[str, Any]] = None,
      **kwargs,
  ) -> ParallelAttentionBlock:
    attn_kwargs = attn_kwargs or {}
    mlp_kwargs = mlp_kwargs or {}
    return cls(
        attention=knn_attn.ImprovedMultiHeadDotProductAttention(
            num_heads=num_heads,
            normalize_qk=normalize_qk,
            **attn_kwargs,
        ),
        mlp=TransformerMLP(
            hidden_size=mlp_size,
            **mlp_kwargs,
        ),
        **kwargs,
    )
