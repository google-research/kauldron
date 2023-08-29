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

"""Miscellaneous Modules."""

from __future__ import annotations

from flax import linen as nn
from kauldron.typing import Array, PRNGKey  # pylint: disable=g-multiple-import
from kauldron.utils import train_property


class Identity(nn.Module):
  """Module that applies the identity function to a single tensor.

  Useful for naming and capturing intermediate variables.
  """

  @nn.compact
  def __call__(self, inputs: Array['...'], *args, **kwargs) -> Array:
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
      self, inputs: Array['*d'], *, rng: PRNGKey | None = None
  ) -> Array['*d']:
    return super().__call__(inputs, deterministic=not self.is_training, rng=rng)
