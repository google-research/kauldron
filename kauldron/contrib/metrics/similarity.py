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

"""Similarity metrics."""

from __future__ import annotations

import dataclasses
from typing import Optional

import flax.struct
import jax.numpy as jnp
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class CosineSimilarity(base.Metric):
  """CosineSimilarity metric."""

  pred: kontext.Key = kontext.REQUIRED
  target: kontext.Key = kontext.REQUIRED
  mask: Optional[kontext.Key] = None
  eps: float = 1e-6

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred: Float["*b n d"],
      target: Float["*b n d"],
      mask: Optional[Bool["*b n 1"] | Float["*b n 1"]] = None,
  ) -> CosineSimilarity.State:
    norm = lambda x: jnp.linalg.norm(x, axis=-1, keepdims=True)
    dot_product = jnp.sum(pred * target, axis=-1, keepdims=True)
    values = dot_product / (norm(pred) * norm(target) + self.eps)
    return self.State.from_values(values=values, mask=mask)
