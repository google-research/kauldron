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

"""Collection of point tracking losses."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from kauldron import kontext
from kauldron.losses import base
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class WeightedL1(base.Loss):
  """Weighted L1 loss."""

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  visible: kontext.Key = kontext.REQUIRED
  bg_mask: kontext.Key = kontext.REQUIRED
  ratio_value: float
  ratio_type: str = "constant"  # "constant" or "dynamic"
  use_bg_ratio_for_occluded: bool = False
  mask_occluded: bool = True

  @typechecked
  def get_values(
      self,
      preds: Float["*a"],
      targets: Float["*a"],
      visible: Float["*b"],
      bg_mask: Float["*b"],  # fg_pixels=1, bg_pixels=0
  ) -> Float["*a"]:
    l1 = jnp.abs(preds - targets)
    if self.mask_occluded:
      l1 *= visible
    if self.use_bg_ratio_for_occluded:
      bg_mask = bg_mask * visible
    fg_l1 = l1 * bg_mask
    if self.ratio_type == "constant":
      ratio = self.ratio_value
    elif self.ratio_type == "dynamic":
      ratio = bg_mask.sum() / (
          jnp.ones_like(bg_mask).sum() - bg_mask.sum() + 1e-6
      )
      ratio = jnp.minimum(ratio, 1e-5)
    else:
      raise ValueError(f"Unsupported ratio type: {self.ratio_type}")
    bg_l1 = (l1 * (1 - bg_mask)) * ratio
    return fg_l1 + bg_l1
