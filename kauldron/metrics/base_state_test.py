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

"""Test the metric base states."""

import flax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Float
import numpy as np
import sklearn.metrics


@flax.struct.dataclass
class AveragePrecision(kd.metrics.AutoState):
  labels: Float['n_samples'] | Float['n_samples n_classes'] = (
      kd.metrics.concat_field()
  )
  logits: Float['n_samples'] | Float['n_samples n_classes'] = (
      kd.metrics.concat_field()
  )

  def compute(self) -> Float['']:
    return sklearn.metrics.average_precision_score(
        self.labels,
        self.logits,
    )


def test_empty_state():
  state = kd.metrics.EmptyState.empty()
  assert not state.compute() and isinstance(state.compute(), dict)
  state1 = state.merge(kd.metrics.EmptyState.empty())
  assert not state1.compute() and isinstance(state1.compute(), dict)


def test_collecting():
  # Examples from
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
  state = AveragePrecision(
      labels=jnp.asarray([0, 0, 1, 1]),
      logits=jnp.asarray([0.1, 0.4, 0.35, 0.8]),
  )

  np.testing.assert_allclose(state.compute(), 0.833333333333333)

  # Merging `empty()` is a no-op
  np.testing.assert_allclose(
      state.merge(AveragePrecision.empty()).finalize().compute(),
      state.finalize().compute(),
  )


def test_collecting_merge():
  # Multi-label not supported with internal sklearn version
  # state0 = AveragePrecision(
  #     labels=jnp.asarray([0, 0, 1, 1]),
  #     logits=jnp.asarray([
  #         [0.7, 0.2, 0.1],
  #         [0.4, 0.3, 0.3],
  #         [0.1, 0.8, 0.1],
  #         [0.2, 0.3, 0.5],
  #     ]),
  # )
  # state1 = AveragePrecision(
  #     labels=jnp.asarray([2, 2]),
  #     logits=jnp.asarray([
  #         [0.4, 0.4, 0.2],
  #         [0.1, 0.2, 0.7],
  #     ]),
  # )
  state0 = AveragePrecision(
      labels=jnp.asarray([0, 0, 1]),
      logits=jnp.asarray([0.1, 0.4, 0.35]),
  )
  state1 = AveragePrecision(
      labels=jnp.asarray([1]),
      logits=jnp.asarray([0.8]),
  )
  final_state = state0.merge(state1).finalize()
  np.testing.assert_allclose(final_state.compute(), 0.833333333333333)
  # Inverse merging should provide the same result
  final_state_2 = state1.merge(state0).finalize()
  np.testing.assert_allclose(final_state_2.compute(), 0.833333333333333)


@flax.struct.dataclass
class FirstNImages(kd.metrics.CollectFirstState):
  images: Float['N h w 3']


def test_collecting_first_image():
  state0 = FirstNImages(images=jnp.zeros((4, 16, 16, 3)), keep_first=5)
  state1 = FirstNImages(images=jnp.ones((4, 16, 16, 3)), keep_first=5)
  final_state = state0.merge(state1).finalize()
  result = final_state.compute()

  assert result.images.shape == (5, 16, 16, 3)
  np.testing.assert_allclose(result.images[:4], np.zeros((4, 16, 16, 3)))
  np.testing.assert_allclose(result.images[4:], np.ones((1, 16, 16, 3)))
  assert isinstance(result.images, np.ndarray)


def test_collecting_first_image_truncate_single_state():
  state = FirstNImages(images=jnp.zeros((23, 16, 16, 3)), keep_first=5)
  result = state.compute()

  assert result.images.shape == (5, 16, 16, 3)
  np.testing.assert_allclose(result.images[:4], jnp.zeros((4, 16, 16, 3)))
  assert isinstance(result.images, np.ndarray)
