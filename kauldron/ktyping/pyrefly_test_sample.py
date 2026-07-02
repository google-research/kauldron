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

"""Sample test file using ktyping annotations for Pyrefly type checking."""

from kauldron.ktyping import Float
from kauldron.ktyping import Shape
from kauldron.ktyping import typechecked


@typechecked
def fn_variadic(images: Float["*b h w c"]) -> Float["*b h w"]:
  return images[..., 0]


@typechecked
def fn_shape(s: Shape["*b t"]) -> Shape["*b t"]:
  return s


@typechecked
def fn_multi_dim(x: Float["b h w c"]) -> Float["b h w c"]:
  return x
