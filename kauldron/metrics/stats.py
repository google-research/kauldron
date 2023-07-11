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

"""Metrics that keep simple statistics about generic values."""
from __future__ import annotations

import dataclasses
from typing import Optional

from clu import metrics as clu_metrics
import flax.struct
import jax.numpy as jnp
from kauldron.metrics import base
from kauldron.typing import Float, Key, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Std(base.Metric):
  """Compute the standard deviation for float values.

  This is a simple example of wrapping a CLU metric.
  """

  values: Key
  mask: Optional[Key] = None

  @flax.struct.dataclass
  class State(clu_metrics.Std):
    pass

  @typechecked
  def get_state(
      self,
      values: Float["*b n"],
      mask: Optional[Float["*b 1"]] = None,
  ) -> Std.State:
    # Note: unlike clu.metrics.Std we support not just batches of scalars but
    # any shape of values. Thus we flatten the values before passing them on.
    values = jnp.ravel(values)
    mask = jnp.ravel(mask) if mask is not None else None
    return self.State.from_model_output(values=values, mask=mask)
