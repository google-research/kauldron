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

"""Metrics for classification tasks."""

from __future__ import annotations

import dataclasses
from typing import Optional

import flax.struct
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class JointAccuracy(base.Metric):
  """Classification Accuracy."""

  logits_a: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels_a: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"
  logits_b: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels_b: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"
  mask: Optional[kontext.Key] = None

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      logits_a: Float["*b n"],
      labels_a: Int["*b 1"],
      logits_b: Float["*b m"],
      labels_b: Int["*b 1"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> JointAccuracy.State:
    correct_a = logits_a.argmax(axis=-1, keepdims=True) == labels_a
    correct_b = logits_b.argmax(axis=-1, keepdims=True) == labels_b
    correct = correct_a & correct_b
    return self.State.from_values(values=correct, mask=mask)
