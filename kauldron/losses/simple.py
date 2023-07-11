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
from kauldron.losses import base
from kauldron.typing import Float, Int, Key, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import optax


# ============================== Distance ===============================


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class L1(base.Loss):
  """L1 loss."""

  preds: Key
  targets: Key

  @typechecked
  def get_values(self, preds: Float["*a"], targets: Float["*a"]) -> Float["*a"]:
    return jnp.abs(preds - targets)


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class L2(base.Loss):
  """L2 loss."""

  preds: Key
  targets: Key

  @typechecked
  def get_values(self, preds: Float["*a"], targets: Float["*a"]) -> Float["*a"]:
    return jnp.square(preds - targets)


# ============================== Classification ===============================


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class SoftmaxCrossEntropy(base.Loss):
  """Softmax cross-entropy loss."""

  logits: Key = "preds.logits"
  labels: Key = "batch.label"

  @typechecked
  def get_values(
      self, logits: Float["*a n"], labels: Float["*a n"]
  ) -> Float["*a 1"]:
    return optax.softmax_cross_entropy(logits, labels)[..., None]


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class SoftmaxCrossEntropyWithIntLabels(base.Loss):
  """Softmax cross-entropy loss with integer labels."""

  logits: Key = "preds.logits"
  labels: Key = "batch.label"

  @typechecked
  def get_values(
      self, logits: Float["*a n"], labels: Int["*a 1"]
  ) -> Float["*a 1"]:
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels.squeeze(-1)
    )[..., None]


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class SigmoidBinaryCrossEntropy(base.Loss):
  """Softmax cross-entropy loss with integer labels."""

  logits: Key = "preds.logits"
  labels: Key = "batch.label"

  @typechecked
  def get_values(self, logits: Float["*a"], labels: Float["*a"]) -> Float["*a"]:
    return optax.sigmoid_binary_cross_entropy(logits, labels)
