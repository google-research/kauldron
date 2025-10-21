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

"""A Summary that reports the confusion matrix in text form."""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional

import einops
import flax.struct
from kauldron import kd
from kauldron.typing import Float, Int, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
from sklearn import metrics as sklearn_metrics  # pylint: disable=g-import-not-at-top


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ConfusionMatrix(kd.metrics.Metric):
  """Confusion Matrix reported in text form.

  Computes the confusion matrix as a numpy array and reports its repr() as a
  text summary.

  Attributes:
    logits: The logits to evaluate.
    labels: The groundtruth labels.
    sample_weight: Sample weights. If None, all samples are considered to have
      weight 1.
    normalize: Normalizes confusion matrix over the true (rows), predicted
      (columns) conditions or all the population. If None, confusion matrix will
      not be normalized.
    precision: The floating point precision to use when formatting the confusion
      matrix. Defaults to 2 significant digits.
  """

  logits: kd.kontext.Key = kd.kontext.REQUIRED  # e.g. "preds.logits"
  labels: kd.kontext.Key = kd.kontext.REQUIRED  # e.g. "batch.label"
  sample_weight: Optional[kd.kontext.Key] = None

  normalize: Literal["true", "pred", "all"] | None = None
  precision: int = 2

  @flax.struct.dataclass
  class State(kd.metrics.AutoState["ConfusionMatrix"]):
    """ConfusionMatrix state."""

    num_classes: int = kd.metrics.static_field()
    labels: Int["b"] = kd.metrics.concat_field()
    preds: Int["b"] = kd.metrics.concat_field()
    sample_weight: Float["b"] | None = kd.metrics.concat_field(default=None)

    @typechecked
    def compute(self) -> str:
      check_type(self.labels, Int["b"])
      check_type(self.preds, Int["b"])
      check_type(self.sample_weight, Float["b"] | None)

      confusion_matrix = sklearn_metrics.confusion_matrix(
          y_true=self.labels,
          y_pred=self.preds,
          sample_weight=self.sample_weight,
          labels=np.arange(self.num_classes),
          normalize=self.parent.normalize,
      )
      with np.printoptions(threshold=np.inf, precision=self.parent.precision):
        return repr(confusion_matrix)

  @typechecked
  def get_state(
      self,
      logits: Float["*b n"],
      labels: Int["*b 1"],
      sample_weight: Optional[Float["*b 1"]] = None,
  ) -> ConfusionMatrix.State:
    logits = einops.rearrange(logits, "... n -> (...) n")
    labels = einops.rearrange(labels, "... -> (...)")
    preds = logits.argmax(axis=-1, keepdims=False)
    if sample_weight is not None:
      sample_weight = einops.rearrange(sample_weight, "... -> (...)")
    return self.State(
        labels=labels,
        preds=preds,
        sample_weight=sample_weight,
        num_classes=logits.shape[-1],
    )
