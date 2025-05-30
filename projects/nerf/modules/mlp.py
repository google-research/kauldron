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

"""MLP."""

from typing import Any

from flax import linen as nn
from projects.nerf.core.typing import ActivationFn  # pylint: disable=g-importing-member


class MLP(nn.Module):
  """MLP."""
  width: int = 256
  num_layers: int = 8
  activation: ActivationFn = nn.relu

  @nn.compact
  def __call__(self, x) -> Any:
    # TODO(epot):
    # * How to choose initialization
    # * Add skip connections
    for _ in range(self.num_layers):
      x = nn.Dense(self.width)(x)
      x = self.activation(x)
    return x
