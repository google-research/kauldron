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

"""Type definitions for Modules."""

from __future__ import annotations
from typing import Callable, Optional, Protocol
from kauldron.typing import Axes, Bool, Float, Initializer, Shape  # pylint: disable=g-multiple-import,g-importing-member,unused-import


ActivationFunction = Callable[[Float['*any']], Float['*any']]


class AttentionModule(Protocol):
  """Interface specification for Attention modules.

  Can be used as type-annotation to specify the interface of a sub-module.

  Examples include: MHDPAttention, ImprovedMHDPAttention.
  """

  num_heads: int

  def __call__(
      self,
      inputs_q: Float['*b q dq'],
      inputs_k: Optional[Float['*b k dk']] = None,  # defaults to inputs_q
      inputs_v: Optional[Float['*b k dv']] = None,  # defaults to inputs_k
      *,
      mask: Optional[Float['*b #num_heads #q #k']] = None,
  ) -> Float['*b q do']:
    pass


class ImageTokenizer(Protocol):
  """Interface for modules that convert images into tokens.

  Module should take a batch of images and return a corresponding batch of
  tokens.

  Examples include Patchify and PatchifyEmbed.
  """

  def __call__(self, image: Float['*b h w c']) -> Float['*b n d']:
    pass


class NormModule(Protocol):
  """Interface specification for norm modules (to be used as type annotation).

  Module should be callable with a single tensor (x) of any shape and return a
  tensor of the same shape.

  Example modules include: nn.LayerNorm, nn.RMSNorm, nn.GroupNorm, nn.BatchNorm.
  """

  def __call__(self, x: Float['*any']) -> Float['*any']:
    pass


class PositionEmbedding(Protocol):
  """Interface specification for position embeddings.

  Examples include: ZeroEmbedding, and LearnedEmbedding.
  """

  def __call__(self, shape: Shape, *, axis: Axes) -> Float['...']:
    pass


class TransformerBlock(Protocol):
  """Interface definition for transformer blocks (for use in type annotations).

  Examples include: PreNormBlock, PostNormBlock and ParallelAttentionBlock.
  """

  def __call__(
      self,
      tokens: Float['*b n d'],
      attention_mask: Bool['*b 1 n n'] | None = None,
  ) -> Float['*b n d']:
    pass
