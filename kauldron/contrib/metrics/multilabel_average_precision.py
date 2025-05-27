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

"""Metrics for mean average precision for multi-label classification tasks."""

from __future__ import annotations

import dataclasses

import flax.struct
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Float, Int, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import sklearn.metrics


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MultilabelAveragePrecision(base.Metric):
  """Multilabel average precision metric.

  This metric is used for multi-label classification tasks.
  """

  scores: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"

  @flax.struct.dataclass
  class State(base_state.CollectingState):
    """MultiLabelAveragePrecision state."""

    scores: Float["*b n"]
    labels: Int["*b n"]

    @typechecked
    def compute(self) -> float:
      out = super().compute()
      labels = out.labels
      check_type(labels, Int["b n"])

      scores = out.scores
      check_type(scores, Float["b n"])
      # Aggregate average precision scores for each class.
      return sklearn.metrics.average_precision_score(
          y_true=labels,
          y_score=scores,
      )

  @typechecked
  def get_state(
      self,
      scores: Float["*b n"],
      labels: Int["*b n"],
  ) -> MultilabelAveragePrecision.State:
    return self.State(
        labels=labels,
        scores=scores,
    )
