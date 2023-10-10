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

"""Metrics for classification tasks."""
from __future__ import annotations

import dataclasses
from typing import Optional

import flax.linen as nn
import flax.struct
import jax.numpy as jnp
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Float, Int, Key, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import sklearn.metrics


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Accuracy(base.Metric):
  """Classification Accuracy."""

  logits: Key = "preds.logits"
  labels: Key = "batch.label"
  mask: Optional[Key] = None

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      logits: Float["*b n"],
      labels: Int["*b 1"],
      mask: Optional[Float["*b 1"]] = None,
  ) -> Accuracy.State:
    correct = logits.argmax(axis=-1, keepdims=True) == labels
    return self.State.from_values(values=correct, mask=mask)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RocAuc(base.Metric):
  """Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""

  logits: Key = "preds.logits"
  labels: Key = "batch.label"
  mask: Optional[Key] = None

  multi_class_mode: str = "ovr"  # One-vs-Rest ("ovr") or One-vs-One ("ovo")

  @flax.struct.dataclass
  class State(base_state.CollectingState):
    labels: Int["*b 1"]
    probs: Float["*b n"]
    mask: Float["*b 1"]

  @typechecked
  def get_state(
      self,
      logits: Float["*b n"],
      labels: Int["*b 1"],
      mask: Optional[Float["*b 1"]] = None,
  ) -> RocAuc.State:
    # simply collect the given values
    mask = jnp.ones_like(labels) if mask is None else mask
    probs = nn.activation.softmax(logits, axis=-1)
    return self.State(
        labels=labels,
        probs=probs,
        mask=mask,
    )

  @typechecked
  def compute(self, state: RocAuc.State) -> float:
    out = state.compute()
    labels = out.labels[..., 0]
    assert isinstance(labels, Int["b"]), labels.shape
    # roc_auc_score is very picky so we first filter out all the classes
    # for which there are no GT examples and renormalize probabilities
    # This will give wrong results, but allows getting a value during training
    # where it cannot be guaranteed that each batch contains all classes.
    unique_labels = np.unique(labels).tolist()
    probs = out.probs[..., unique_labels]
    probs /= probs.sum(axis=-1, keepdims=True)  # renormalize
    assert isinstance(probs, Float["b n"]), probs.shape
    mask = out.mask[..., 0].astype(np.float32)
    assert isinstance(mask, Float["b"]), mask.shape

    return sklearn.metrics.roc_auc_score(
        y_true=labels,
        y_score=probs,
        sample_weight=mask,
        labels=unique_labels,
        multi_class=self.multi_class_mode,
    )
