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

"""Test."""

import chex
import flax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Float
import numpy as np
import sklearn


@flax.struct.dataclass
class AveragePrecision(kd.metrics.CollectingState):
  labels: Float['n_samples'] | Float['n_samples n_classes']
  logits: Float['n_samples'] | Float['n_samples n_classes']

  def compute(self) -> Float['']:
    values = super().compute()
    return sklearn.metrics.average_precision_score(
        values.labels,
        values.logits,
    )


def test_collecting():
  # Examples from
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
  state = AveragePrecision(
      labels=jnp.asarray([0, 0, 1, 1]),
      logits=jnp.asarray([0.1, 0.4, 0.35, 0.8]),
  )

  np.testing.assert_allclose(state.compute(), 0.833333333333333)

  # Merging `empty()` is a no-op
  chex.assert_trees_all_close(state, state.merge(AveragePrecision.empty()))


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
  final_state = state0.merge(state1)
  np.testing.assert_allclose(final_state.compute(), 0.833333333333333)
  # Inverse merging should provide the same result
  np.testing.assert_allclose(state1.merge(state0).compute(), 0.833333333333333)


@flax.struct.dataclass
class FirstNImages(kd.metrics.CollectFirstState):
  images: Float['N h w 3']


def test_collecting_first_image():
  state0 = FirstNImages(images=jnp.zeros((4, 16, 16, 3)), keep_first=5)
  state1 = FirstNImages(images=jnp.ones((4, 16, 16, 3)), keep_first=5)
  final_state = state0.merge(state1)
  result = final_state.compute()

  assert result.images.shape == (5, 16, 16, 3)
  np.testing.assert_allclose(result.images[:4], jnp.zeros((4, 16, 16, 3)))
  np.testing.assert_allclose(result.images[4:], jnp.ones((1, 16, 16, 3)))
