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

"""Metrics for classification tasks."""

from __future__ import annotations

import dataclasses
from typing import Optional

import flax.linen as nn
import flax.struct
import jax.numpy as jnp
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, Int, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import sklearn.metrics


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Accuracy(base.Metric):
  """Classification Accuracy."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"
  mask: Optional[kontext.Key] = None

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      logits: Float["*b n"],
      labels: Int["*b 1"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> Accuracy.State:
    correct = logits.argmax(axis=-1, keepdims=True) == labels
    return self.State.from_values(values=correct, mask=mask)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Precision1(base.Metric):
  """Precision@1 for multilabel classification."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.labels"
  mask: Optional[kontext.Key] = None

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      logits: Float["*b n"],
      labels: Float["*b n"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> Precision1.State:
    pred_argmax = logits.argmax(axis=-1, keepdims=True)
    correct = jnp.take_along_axis(labels, pred_argmax, -1)
    return self.State.from_values(values=correct, mask=mask)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class BinaryAccuracy(base.Metric):
  """Classification Accuracy for Binary classification tasks."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      logits: Float["*any"],
      labels: Int["*any"],
      mask: Optional[Bool["*#any"] | Float["*#any"]] = None,
  ):
    correct = (logits > 0) == labels
    return self.State.from_values(values=correct, mask=mask)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RocAuc(base.Metric):
  """Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"
  mask: Optional[kontext.Key] = None

  multi_class_mode: str = "ovr"  # One-vs-Rest ("ovr") or One-vs-One ("ovo")

  @flax.struct.dataclass
  class State(base_state.CollectingState):
    """RocAuc state."""

    labels: Int["*b 1"]
    probs: Float["*b n"]
    mask: Bool["*b 1"] | Float["*b 1"]

    @typechecked
    def compute(self) -> float:
      out = super().compute()
      labels = out.labels[..., 0]
      check_type(labels, Int["b"])
      # roc_auc_score is very picky so we first filter out all the classes
      # for which there are no GT examples and renormalize probabilities
      # This will give wrong results, but allows getting a value during training
      # where it cannot be guaranteed that each batch contains all classes.
      unique_labels = np.unique(labels).tolist()
      probs = out.probs[..., unique_labels]
      probs /= probs.sum(axis=-1, keepdims=True)  # renormalize
      check_type(probs, Float["b n"])
      mask = out.mask[..., 0].astype(np.float32)
      check_type(mask, Float["b"])

      return sklearn.metrics.roc_auc_score(
          y_true=labels,
          y_score=probs,
          sample_weight=mask,
          labels=unique_labels,
          multi_class=self.parent.multi_class_mode,
      )

  @typechecked
  def get_state(
      self,
      logits: Float["*b n"],
      labels: Int["*b 1"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> RocAuc.State:
    # simply collect the given values
    mask = jnp.ones_like(labels) if mask is None else mask
    probs = nn.activation.softmax(logits, axis=-1)
    return self.State(
        labels=labels,
        probs=probs,
        mask=mask,
    )
