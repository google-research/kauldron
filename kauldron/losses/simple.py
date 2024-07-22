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

"""Collection of standard losses."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from kauldron import kontext
from kauldron.losses import base
from kauldron.typing import Array, Float, Int, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import optax


# ============================== Values ===============================


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Value(base.Loss):
  """Value loss."""

  values: kontext.Key = kontext.REQUIRED

  @typechecked
  def get_values(self, values: Float["*a"]) -> Float["*a"]:
    return values


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class AbsoluteValue(base.Loss):
  """Absolute value loss."""

  values: kontext.Key = kontext.REQUIRED

  @typechecked
  def get_values(self, values: Float["*a"]) -> Float["*a"]:
    return jnp.abs(values)


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


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class Huber(base.Loss):
  """Huber loss."""
  delta: float = 1.0

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED

  @typechecked
  def get_values(self, preds: Float["*a"], targets: Float["*a"]) -> Float["*a"]:
    l2_term = 0.5 * jnp.square(preds - targets)
    l1_term = self.delta * (jnp.abs(preds - targets) - 0.5 * self.delta)
    return jnp.where(jnp.abs(preds - targets) < self.delta, l2_term, l1_term)


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class NegativeCosineSimilarity(base.Loss):
  """Negative Cosine Similarity loss."""

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  eps: float = 1e-8

  def _safe_normalize(self, x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + self.eps)

  @typechecked
  def get_values(
      self,
      preds: Float["*a c"],
      targets: Float["*a c"],
  ) -> Float["*a 1"]:
    preds = self._safe_normalize(preds)
    targets = self._safe_normalize(targets)

    similarity = jnp.sum(preds * targets, axis=-1, keepdims=True)
    return - similarity

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
class SigmoidBinaryCrossEntropy(base.Loss):
  """Sigmoid cross-entropy loss with binary labels."""

  logits: kontext.Key = kontext.REQUIRED  # e.g. "preds.logits"
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.label"

  @typechecked
  def get_values(
      self, logits: Float["*a n"], labels: Array["*a n"]
  ) -> Float["*a n"]:
    return optax.sigmoid_binary_cross_entropy(logits, labels)
