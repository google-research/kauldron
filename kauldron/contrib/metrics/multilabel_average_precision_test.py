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

"""Test."""

from jax import numpy as jnp
from kauldron.contrib.metrics import multilabel_average_precision
import numpy as np
import pytest


@pytest.mark.parametrize(('num_samples', 'num_classes'), [(3, 3), (8, 6)])
def test_multilabel_average_precision_correct_classes(
    num_samples: int, num_classes: int
):
  """Verifies that mAP is 1 when scores match labels for all samples."""
  metric = multilabel_average_precision.MultilabelAveragePrecision()

  if num_samples < num_classes:
    raise ValueError(
        f'num_samples {num_samples} must be less than num_classes'
        f' {num_classes} to reach full average precision score.'
    )
  scores = jnp.eye(num_samples, num_classes, dtype=jnp.float32)
  labels = jnp.eye(num_samples, num_classes, dtype=jnp.int32)

  s0 = metric.get_state(scores=scores, labels=labels)

  ref_correct_score = s0.compute()
  assert ref_correct_score == 1
  s1 = multilabel_average_precision.MultilabelAveragePrecision().empty()

  # Merging empty states when added before or after should not change the final
  # score.
  x = s0.merge(s1).finalize().compute()
  y = s1.merge(s0).finalize().compute()  # pylint: disable=arguments-out-of-order
  np.testing.assert_allclose(x, ref_correct_score)
  np.testing.assert_allclose(y, ref_correct_score)

  # Accuracy should remain 1 when combining twice scores with matching labels.
  s0 = metric.get_state(scores=scores, labels=labels)
  s1 = metric.get_state(scores=scores, labels=labels)
  mean_ap = s0.merge(s1).finalize().compute()
  np.testing.assert_allclose(mean_ap, 1)


@pytest.mark.parametrize(
    ['scores', 'labels', 'expected_map'],
    [
        (
            jnp.asarray([
                [0.9, 0.6, 0.1, 0.9, 0.9],
                [1.0, 0.6, 0.0, 0.0, 0.9],
            ]),
            jnp.asarray([
                [1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
            ]),
            0.5,
        ),
        # Example from torchmetrics MultilabelAveragePrecision.
        # https://lightning.ai/docs/torchmetrics/stable/classification/average_precision.html#multilabelaverageprecision
        (
            jnp.asarray([
                [0.75, 0.05, 0.35],
                [0.45, 0.75, 0.05],
                [0.05, 0.55, 0.75],
                [0.05, 0.65, 0.05],
            ]),
            jnp.asarray([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 1]]),
            0.75,
        ),
    ],
)
def test_multilabel_average_precision(
    scores: jnp.ndarray, labels: jnp.ndarray, expected_map: float
):
  metric = multilabel_average_precision.MultilabelAveragePrecision()
  state = metric.get_state(scores=scores, labels=labels)
  mean_ap = state.compute()
  np.testing.assert_allclose(mean_ap, expected_map)
