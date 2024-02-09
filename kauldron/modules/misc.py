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

"""Miscellaneous Modules."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Literal, Optional

import einops
from flax import linen as nn
import flax.core
import jax.numpy as jnp
from kauldron import kontext
from kauldron.typing import Array, Float, PRNGKey, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import train_property

FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict


class Identity(nn.Module):
  """Module that applies the identity function to a single tensor.

  Useful for naming and capturing intermediate variables.
  """

  @nn.compact
  def __call__(self, inputs: Array['*any'], *args, **kwargs) -> Array['*any']:
    return inputs


class Dropout(nn.Dropout):
  """Wrapper around `nn.Dropout` but using `kd.nn.train_property`."""

  is_training = train_property.train_property()

  def __post_init__(self):
    super().__post_init__()
    if self.deterministic is not None:
      raise ValueError(
          '`kd.nn.Dropout` should not use `deterministic`. Instead the '
          'training mode is set through `is_training_property`. See '
          '`kd.nn.train_property`.'
      )

  @nn.compact
  def __call__(  # pytype: disable=signature-mismatch
      self, inputs: Array['*any'], *, rng: PRNGKey | None = None
  ) -> Array['*any']:
    return super().__call__(inputs, deterministic=not self.is_training, rng=rng)


class Rearrange(nn.Module):
  """Wrapper around `einops.rearrange` for usage e.g. in `nn.Sequential`.

  Example:

  ```
  cfg.model = kd.nn.Sequential(
      inputs="batch.image",
      layers=[
          nn.Conv(features=192, kernel_size=(8, 8), strides=(8, 8)),
          # flatten the image dimensions before applying the transformer blocks
          kd.nn.Rearrange(pattern="... h w d -> ... (h w) d"),
          kd.nn.PreNormBlock.from_spec(num_heads=12),
          ...
      ]
  )
  ```

  Attributes:
    pattern: `einops.rearrange` pattern, e.g. "b h w c -> b c (h w)"
    axes_lengths: a dictionary for specifying additional axis sizes that cannot
      be inferred from the pattern and the tensor alone.
  """

  pattern: str
  axes_lengths: dict[str, int] = dataclasses.field(default_factory=FrozenDict)

  @typechecked
  def __call__(self, tensor: Array['...']) -> Array['...']:
    return einops.rearrange(tensor, pattern=self.pattern, **self.axes_lengths)


class Reduce(nn.Module):
  """Wrapper around `einops.reduce` for usage e.g. in `nn.Sequential`.

  Example:

  ```
  cfg.model = kd.nn.Sequential(
      inputs="batch.image",
      layers=[
          ...
          kd.nn.PreNormBlock.from_spec(num_heads=12),
          # use Reduce to implement Global Average Pooling
          kd.nn.Reduce(pattern="... n d -> ... d", reduction="mean"),
          nn.Dense(features=1000),
      ]
  )
  ```

  Attributes:
    pattern: `einops.reduce` pattern, e.g. "b h w c -> b c"
    reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod')
    axes_lengths: a dictionary for specifying additional axis sizes that cannot
      be inferred from the pattern and the tensor alone.
  """

  pattern: str
  reduction: Literal['min', 'max', 'sum', 'mean', 'prod']
  axes_lengths: dict[str, int] = dataclasses.field(default_factory=FrozenDict)

  @typechecked
  def __call__(self, tensor: Array['...']) -> Array['...']:
    return einops.reduce(
        tensor,
        pattern=self.pattern,
        reduction=self.reduction,
        **self.axes_lengths,
    )


class DummyModel(nn.Module):
  """Empty model that ignores inputs and always produces a single logit of 42.

  Can be useful as a placeholder model in a config, while working on and testing
  other parts such as the data pipeline.
  """

  inputs: Optional[kontext.Key] = None

  @typechecked
  def __call__(self, inputs: Any = None) -> dict[str, Float['1']]:
    return {'logits': jnp.ones((1,)) * 42}
