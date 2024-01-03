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

"""Collection of standard losses."""
from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from kauldron import kontext
from kauldron.losses import base
from kauldron.typing import Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import optax


# ============================== Distance ===============================


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class L1(base.Loss):
  """L1 loss."""

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED

  @typechecked
  def get_values(self, preds: Float["*a"], targets: Float["*a"]) -> Float["*a"]:
    return jnp.abs(preds - targets)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class L2(base.Loss):
  """L2 loss."""

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED

  @typechecked
  def get_values(self, preds: Float["*a"], targets: Float["*a"]) -> Float["*a"]:
    return jnp.square(preds - targets)


# ============================== Classification ===============================


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class SoftmaxCrossEntropy(base.Loss):
  """Softmax cross-entropy loss."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"

  @typechecked
  def get_values(
      self, logits: Float["*a n"], labels: Float["*a n"]
  ) -> Float["*a 1"]:
    return optax.softmax_cross_entropy(logits, labels)[..., None]


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class SoftmaxCrossEntropyWithIntLabels(base.Loss):
  """Softmax cross-entropy loss with integer labels."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"

  @typechecked
  def get_values(
      self, logits: Float["*a n"], labels: Int["*a 1"]
  ) -> Float["*a 1"]:
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels.squeeze(-1)
    )[..., None]


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class SingleClassSigmoidBinaryCrossEntropy(base.Loss):
  """Sngle-class sigmoid cross-entropy loss with binary integer labels."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"

  @typechecked
  def get_values(
      self, logits: Float["*a 1"], labels: Int["*a 1"]
  ) -> Float["*a n"]:
    # Optax uses a single implementation for both single & multi-class, but
    # always expects float inputs.
    labels = labels.astype(jnp.float32)
    return optax.sigmoid_binary_cross_entropy(logits, labels)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class MultiClassSigmoidBinaryCrossEntropy(base.Loss):
  """Sigmoid cross-entropy loss with multi-class float labels."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"

  @typechecked
  def get_values(
      self, logits: Float["*a n"], labels: Float["*a n"]
  ) -> Float["*a n"]:
    return optax.sigmoid_binary_cross_entropy(logits, labels)
