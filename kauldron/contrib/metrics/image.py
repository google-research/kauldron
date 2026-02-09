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

"""PSNR which resizes the prediction to the target if there is a mismatch."""

import dataclasses
from typing import Optional
import jax
from kauldron import kd
from kauldron.typing import Bool, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PsnrWithResize(kd.metrics.Psnr):
  """PSNR which resizes the prediction to the target if there is a mismatch."""

  @typechecked
  def get_state(
      self,
      pred: Float["*b h_p w_p c"],
      target: Float["*b h_t w_t c"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> kd.metrics.Psnr.State:
    if pred.shape[1] != target.shape[1] or pred.shape[2] != target.shape[2]:
      pred = jax.image.resize(
          image=pred,
          shape=(
              pred.shape[0],
              target.shape[1],
              target.shape[2],
              pred.shape[3],
          ),
          method=jax.image.ResizeMethod.LINEAR,
      )
    return super().get_state(pred, target, mask)
