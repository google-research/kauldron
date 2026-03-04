# Copyright 2026 The kauldron Authors.
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

"""Examples for KdNnxModule."""

import dataclasses
from flax import nnx
from kauldron import kontext
from kauldron.contrib.modules import knnx


class MyNnxModule(nnx.Module):
  """My nnx module. This is a regular nnx module."""

  def __init__(
      self, input_dim: int = 3, hdim: int = 10, rngs: nnx.Rngs = nnx.Rngs(0)
  ):
    self.input_dim = input_dim
    self.hdim = hdim
    self.lin = nnx.Linear(input_dim, hdim, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)

  def __call__(self, x):
    x = self.lin(x)
    x = self.dropout(x)
    x = nnx.relu(x)
    return x


@dataclasses.dataclass(kw_only=True)
class SimpleKdNnxModule(knnx.KdNnxModule):
  """Simple nnx module."""

  input_dim: int = 3
  hdim: int = 128
  output_dim: int = 1000

  image: kontext.Key = "batch.image"

  def setup(self, rngs: nnx.Rngs = nnx.Rngs(0)):
    self.backbone = MyNnxModule(
        input_dim=self.input_dim, hdim=self.hdim, rngs=rngs
    )
    self.proj = nnx.Linear(self.hdim, self.output_dim, rngs=rngs)

  def __call__(self, image):
    x = self.backbone(image)
    x = self.proj(x)
    return x
