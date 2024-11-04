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

"""Map transforms."""

import dataclasses
import typing
from typing import Any

import einops
from etils import enp
import flax.core
from kauldron.data.transforms import base
from kauldron.typing import XArray, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np

_FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Rearrange(base.ElementWiseTransform):
  """Einops rearrange on a single element.

  Mostly a wrapper around einops.rearrange, but also supports basic types like
  int, float, lists and tuples (which are converted to a numpy array first).

  Example:

  ```
  cfg.train_ds = kd.data.Tfds(
      ...
      transforms=[
          ...,
          kd.data.Rearrange(key="image", pattern="h w c -> (h w c)"),
      ]
  )
  ```

  Attributes:
    pattern: `einops.rearrange` pattern, e.g. "b h w c -> b c (h w)"
    axes_lengths: a dictionary for specifying additional axis sizes that cannot
      be inferred from the pattern and the tensor alone.
  """

  pattern: str
  axes_lengths: dict[str, int] = dataclasses.field(default_factory=_FrozenDict)

  @typechecked
  def map_element(self, element: Any) -> XArray:
    # Ensure element is an array (and not a python builtin)
    # This is useful e.g. for pygrain pipelines because often "label" will be
    # int and not an array, yet one might want to reshape it.
    xnp = enp.lazy.get_xnp(element, strict=False)
    element = xnp.asarray(element)

    return einops.rearrange(element, self.pattern, **self.axes_lengths)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ValueRange(base.ElementWiseTransform):
  """Map the value range of an element."""

  vrange: tuple[float, float]
  in_vrange: tuple[float, float] = (0.0, 255.0)

  dtype: Any = np.float32
  clip_values: bool = True

  @typechecked
  def map_element(self, element: XArray["*any"]) -> XArray["*any"]:
    xnp = enp.lazy.get_xnp(element)
    dtype = enp.lazy.as_np_dtype(self.dtype)
    element = xnp.asarray(element, dtype=dtype)
    in_min, in_max = self.in_vrange
    out_min, out_max = self.vrange
    element = (element - in_min) / (in_max - in_min)
    element = element * (out_max - out_min) + out_min
    if self.clip_values:
      element = xnp.clip(element, out_min, out_max)
    return element


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Gather(base.ElementWiseTransform):
  """Gathers entries along a single dimension."""

  axis: int
  indices: tuple[int, ...]

  @typechecked
  def map_element(self, element: XArray) -> XArray:
    xnp = enp.lazy.get_xnp(element)
    return xnp.take(element, self.indices, axis=self.axis)
