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

"""A no-op training step, that can be used for benchmarking of data pipelines."""

import dataclasses
import functools
from typing import Any
import flax.core
import jax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import PyTree  # pylint: disable=g-importing-member
from kauldron.utils import utils

TrainState = kd.train.TrainState


class NoopTrainStep(kd.train.TrainStep):
  """No-op training step.

  Useful for debugging  / benchmarking data pipelines.
  """

  @utils.checkify_ignore
  @functools.partial(
      jax.jit,
      static_argnames=("self", "elem_spec", "model_method"),
  )
  def _init_model(self, elem_spec, **kwargs):
    state = kd.train.TrainState(  # pytype: disable=wrong-arg-types
        step=jnp.asarray(0),
        params=flax.core.FrozenDict(),
        opt_state=flax.core.FrozenDict(),
        collections=flax.core.FrozenDict(),
    )
    return kd.sharding.with_sharding_constraint(state, self.sharding.state)  # pytype: disable=wrong-arg-types

  def _init_transform(self, state: TrainState) -> TrainState:
    return state

  def _init_transform_after_optimizer(self, state: TrainState) -> TrainState:
    return state

  @utils.checkify_ignore
  @functools.partial(
      jax.jit,
      static_argnames="self",
      donate_argnames=("state",),
  )
  def _init_optimizer(
      self,
      state: TrainState,
  ) -> TrainState:
    return kd.sharding.with_sharding_constraint(state, self.sharding.state)  # pytype: disable=wrong-arg-types

  def _step(
      self,
      state: TrainState,
      batch: PyTree[Any],
  ) -> tuple[TrainState, kd.train.Context]:
    context = kd.train.Context.from_state_and_batch(state=state, batch=batch)
    context = dataclasses.replace(
        context,
        loss_states=flax.core.FrozenDict(),
        metric_states=flax.core.FrozenDict(),
        summary_states=flax.core.FrozenDict(),
    )
    return state, context
