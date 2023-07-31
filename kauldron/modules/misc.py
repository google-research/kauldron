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
from kauldron.typing import Array


class Identity(nn.Module):
  """Module that applies the identity function to a single tensor.

  Useful for naming and capturing intermediate variables.
  """

  @nn.compact
  def __call__(self, inputs: Array["..."], *args, **kwargs) -> Array:
    return inputs
