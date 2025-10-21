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

"""Test the confusion matrix summary."""

from jax import numpy as jnp
from kauldron.contrib.summaries import confusion_matrix
import numpy as np


def test_confusion_matrix():
  cm = confusion_matrix.ConfusionMatrix()
  logits = jnp.asarray([
      [0.1, 0.8, 0.1],  # pred 1
      [0.8, 0.1, 0.1],  # pred 0
      [0.1, 0.1, 0.8],  # pred 2
  ])
  labels = jnp.asarray([[1], [0], [2]], dtype=jnp.int32)
  state = cm.get_state(logits=logits, labels=labels)
  cm = state.finalize().compute()
  assert cm == repr(np.eye(3, dtype=int))


def test_confusion_matrix_merge():
  cm = confusion_matrix.ConfusionMatrix()
  logits1 = jnp.asarray([[0.9, 0.1], [0.1, 0.9]])  # pred 0, 1
  labels1 = jnp.asarray([[0], [0]], dtype=jnp.int32)  # label 0, 0
  state1 = cm.get_state(logits=logits1, labels=labels1)

  logits2 = jnp.asarray([[0.1, 0.9], [0.1, 0.9]])  # pred 1, 1
  labels2 = jnp.asarray([[1], [0]], dtype=jnp.int32)  # label 1, 0
  state2 = cm.get_state(logits=logits2, labels=labels2)

  expected_cm = np.array([[1, 2], [0, 1]])
  state_merged = state1.merge(state2)
  cm = state_merged.finalize().compute()
  assert cm == repr(expected_cm)

  cm_norm_true = confusion_matrix.ConfusionMatrix(normalize="true")
  state1_norm = cm_norm_true.get_state(logits=logits1, labels=labels1)
  state2_norm = cm_norm_true.get_state(logits=logits2, labels=labels2)
  cm_norm_true = state1_norm.merge(state2_norm).finalize().compute()
  expected_cm_norm_true = expected_cm / expected_cm.sum(axis=1, keepdims=True)
  with np.printoptions(threshold=np.inf, precision=2):
    assert cm_norm_true == repr(expected_cm_norm_true)


def test_confusion_matrix_mask():
  cm = confusion_matrix.ConfusionMatrix()
  logits = jnp.asarray([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])  # pred 0, 1, 0
  labels = jnp.asarray([[0], [1], [0]], dtype=jnp.int32)  # label 0, 1, 0
  sample_weight = jnp.asarray(
      [[1], [1], [0]], dtype=jnp.float32
  )  # mask out last sample
  state = cm.get_state(
      logits=logits, labels=labels, sample_weight=sample_weight
  )
  cm = state.finalize().compute()
  # labels=[0,1], preds=[0,1]
  expected_cm = np.array([[1.0, 0.0], [0.0, 1.0]])
  assert cm == repr(expected_cm)
