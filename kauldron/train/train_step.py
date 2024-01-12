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

"""Default TrainState and TrainStep implementations."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

from etils import epy
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron import losses as kd_losses
from kauldron import metrics as kd_metrics
from kauldron import summaries as kd_summaries
from kauldron.checkpoints import partial_loader
import kauldron.data.utils as data_utils
from kauldron.train import rngs_lib
from kauldron.typing import ElementSpec, Float, PyTree  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import config_util
from kauldron.utils import context as context_lib
from kauldron.utils import jax_utils
from kauldron.utils import train_property  # pylint: disable=unused-import
from kauldron.utils.sharding_utils import sharding  # pylint: disable=g-importing-member
import optax

_Params = PyTree[Float["..."]]


@flax.struct.dataclass
class TrainState:
  """Data structure for checkpointing the model."""

  _: dataclasses.KW_ONLY

  step: int

  params: Optional[_Params]
  opt_state: Optional[PyTree[Float["..."]]]

  training_time_hours: float

  def next(self, new_params=None, new_opt_state=None) -> TrainState:
    step = self.step + 1
    new_params = new_params or self.params
    new_opt_state = new_opt_state or self.opt_state
    return self.replace(
        step=step,
        params=new_params,
        opt_state=new_opt_state,
    )

  def replace(self, **changes: Any) -> TrainState:
    return dataclasses.replace(self, **changes)


@flax.struct.dataclass
class Auxiliaries:
  """Auxiliaries (intermediate states to be accumulated)."""

  loss_states: Mapping[str, kd_metrics.State] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  metric_states: Mapping[str, kd_metrics.State] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  summary_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )

  def replace(self, **changes: Any) -> Auxiliaries:
    return dataclasses.replace(self, **changes)

  def merge(self, other: Optional[Auxiliaries]) -> Auxiliaries:
    """Accumulate auxiliary."""
    if other is None:
      return self
    # TODO(epot): How to merge summaries ?
    return self.replace(
        loss_states=_reduce_states(self.loss_states, other.loss_states),
        metric_states=_reduce_states(self.metric_states, other.metric_states),
    )

  def __or__(self, other: Auxiliaries | None) -> Auxiliaries:
    """Alias for `.merge()`: `aux = aux1 | aux2`."""
    if other is None:
      return self
    return self.merge(other)

  def __ror__(self, other: Auxiliaries | None) -> Auxiliaries:
    """Alias for `.merge()`: `aux = aux1 | aux2`."""
    if other is None:
      return self
    return other.merge(self)

  def compute(self, *, flatten: bool = True) -> AuxiliariesOutput:
    """Compute losses and metrics."""
    # train losses
    loss_values = jax.tree_map(
        _compute_metric, self.loss_states, is_leaf=kd_metrics.State.isinstance
    )
    # Multi-process communication
    with jax.spmd_mode("allow_all"), jax.transfer_guard("allow"):
      total_loss = jnp.sum(jnp.asarray(list(loss_values.values())))

    if not isinstance(loss_values, dict):
      loss_values = dict(loss_values)  # Convert FrozenDict, ImmutableDict
    loss_values["total"] = total_loss

    # train metrics
    metric_values = jax.tree_map(
        _compute_metric, self.metric_states, is_leaf=kd_metrics.State.isinstance
    )

    if flatten:
      metric_values = kontext.flatten_with_path(
          metric_values, prefix="metrics", separator="/"
      )
      loss_values = kontext.flatten_with_path(
          loss_values, prefix="losses", separator="/"
      )

    return AuxiliariesOutput(
        loss_values=loss_values,
        metric_values=metric_values,
        # hist_summaries=hist_summaries,
        # image_summaries=image_summaries,
    )


def _compute_metric(state: Any):
  """Compute the value of a metric for a given state and return the result."""
  # Accept cross-process computation (some metrics cannot be jitted)
  with jax.spmd_mode("allow_all"), jax.transfer_guard("allow"):
    return state.compute()


@flax.struct.dataclass
class AuxiliariesOutput:
  """Auxiliaries final values (after merge and compute)."""

  loss_values: dict[str, Any] = dataclasses.field(default_factory=dict)
  metric_values: dict[str, Any] = dataclasses.field(default_factory=dict)
  # TODO(klausg): Add summaries


def _reduce_states_single(
    states: tuple[kd_metrics.State, ...]
) -> kd_metrics.State:
  final_state, *rest_states = states
  for state in rest_states:
    final_state = final_state.merge(state)
  return final_state


def _reduce_states(
    *all_states: Mapping[str, kd_metrics.State]
) -> dict[str, kd_metrics.State]:
  """Merge all the states from the different metrics."""
  return {
      k: _reduce_states_single(states)
      for k, states in epy.zip_dict(*all_states)
  }


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class ModelWithAux(config_util.UpdateFromRootCfg):
  """Wrapper around model which also compute the summaries and metrics."""

  model: nn.Module = config_util.ROOT_CFG_REF.model
  losses: Mapping[str, kd_losses.Loss] = config_util.ROOT_CFG_REF.train_losses
  metrics: Mapping[str, kd_metrics.Metric] = (
      config_util.ROOT_CFG_REF.train_metrics
  )
  summaries: Mapping[str, kd_summaries.Summary] = (
      config_util.ROOT_CFG_REF.train_summaries
  )

  def init(  # pylint:disable=missing-function-docstring
      self,
      init_rngs: rngs_lib.Rngs,
      elem_spec: ElementSpec,
      model_method: Optional[str] = None,
  ) -> _Params:
    self._assert_root_cfg_resolved()
    mock_batch = data_utils.mock_batch_from_elem_spec(elem_spec)
    context = context_lib.Context(step=0, batch=mock_batch)
    args, kwargs = data_utils.get_model_inputs(self.model, context)
    params = self.model.init(
        init_rngs,
        *args,
        method=model_method,
        is_training_property=True,
        **kwargs,
    )["params"]
    params = flax.core.unfreeze(params)
    return params

  @jax.named_call
  def forward(
      self,
      params,
      *,
      batch,
      rngs: rngs_lib.Rngs,
      step: int,
      is_training: bool,
  ) -> tuple[float, context_lib.Context]:
    """Forward pass of the model including losses."""
    context = context_lib.Context(step=step, batch=batch, params=params)
    args, kwargs = data_utils.get_model_inputs(self.model, context)
    preds, intermediates = self.model.apply(  # TODO(klausg): capture mutables?
        {"params": params},
        *args,
        rngs=rngs,
        capture_intermediates=True,  # TODO(klausg): check if need a filter here
        is_training_property=is_training,
        **kwargs,
    )
    context = context.replace(
        preds=preds, interms=intermediates["intermediates"]
    )
    loss_total, loss_states = kd_losses.compute_losses(
        losses=self.losses, context=context
    )
    return loss_total, context.replace(
        loss_states=loss_states,
        loss_total=loss_total,
    )

  @jax.named_call
  def get_aux(
      self,
      context: context_lib.Context,
      *,
      # TODO(epot): Better signature
      return_losses: bool = False,
      return_metrics: bool = False,
      return_summaries: bool = False,
  ) -> Auxiliaries:
    """Get auxilaries."""
    aux = Auxiliaries()
    if return_losses:
      aux = aux.replace(loss_states=context.loss_states)

    if return_metrics:
      aux = aux.replace(
          metric_states=jax.tree_util.tree_map(
              lambda m: m.get_state_from_context(context), self.metrics
          )
      )

    if return_summaries:
      aux = aux.replace(
          summary_kwargs={
              k: summary.gather_kwargs(context)
              for k, summary in self.summaries.items()
          }
      )
    return aux


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class TrainStep(config_util.UpdateFromRootCfg):
  """Training Step."""

  model_with_aux: ModelWithAux = dataclasses.field(default_factory=ModelWithAux)
  optimizer: optax.GradientTransformation = config_util.ROOT_CFG_REF.optimizer
  rng_streams: rngs_lib.RngStreams = config_util.ROOT_CFG_REF.rng_streams
  init_transforms: Mapping[str, partial_loader.AbstractPartialLoader] = (
      config_util.ROOT_CFG_REF.init_transforms
  )

  def update_from_root_cfg(self, root_cfg) -> TrainStep:
    new_self = super().update_from_root_cfg(root_cfg)
    assert isinstance(new_self, TrainStep)
    new_self = dataclasses.replace(
        new_self,
        model_with_aux=new_self.model_with_aux.update_from_root_cfg(root_cfg),
    )
    return new_self

  def init(
      self,
      elem_spec: ElementSpec,
      *,
      model_method: Optional[str] = None,
      skip_transforms: bool = False,
  ) -> TrainState:
    """Initialize the model and return the initial TrainState.

    Args:
      elem_spec: Structure of the input batch
      model_method: Name of the flax model method (default to `__call__`)
      skip_transforms: If `False`, apply the `init_transform` on the state (e.g.
        to overwrite the weights with ones from another checkpoint).

    Returns:
      state: The training state
    """
    self._assert_root_cfg_resolved()
    if isinstance(elem_spec, dict):
      elem_spec = flax.core.freeze(elem_spec)
    state = self._init_model(elem_spec, model_method=model_method)
    if not skip_transforms:
      # if restoring a checkpoint we can skip the (potentially slow) transforms
      state = self._init_transforms(state)
    # state = sharding.device_put(state, sharding.REPLICATED)
    state = self._init_optimizer(state)
    return state

  @jax_utils.jit(
      out_shardings=lambda: sharding.REPLICATED,
      static_argnames=("self", "elem_spec", "model_method"),
  )
  def _init_model(
      self,
      elem_spec: ElementSpec,
      *,
      model_method: Optional[str] = None,
  ) -> TrainState:
    """Initialize the model and return the initial TrainState."""
    params = self.model_with_aux.init(
        self.rng_streams.init_rngs(), elem_spec, model_method=model_method
    )
    return TrainState(
        step=0,
        params=params,
        opt_state=None,
        training_time_hours=0.0,
    )

  def _init_transforms(self, state: TrainState) -> TrainState:
    """Run any additional init transformations and return the updated state."""
    for init_transf in self.init_transforms.values():
      state = init_transf.transform(state)
    return state

  @jax_utils.jit(
      out_shardings=lambda: sharding.REPLICATED,
      static_argnames="self",
  )
  def _init_optimizer(
      self,
      state: TrainState,
  ) -> TrainState:
    """Initialize the model and return the initial TrainState."""
    opt_state = self.optimizer.init(state.params)
    return state.replace(opt_state=opt_state)

  @jax_utils.jit(
      static_argnames=(
          "self",
          "return_losses",
          "return_metrics",
          "return_summaries",
      ),
      donate_argnames=("state",),
      # in_shardings=lambda: dict(  # pylint: disable=g-long-lambda
      #     state=sharding.REPLICATED,
      #     batch=sharding.SHARDED,
      # ),
      out_shardings=lambda: sharding.REPLICATED,
  )
  @jax.named_call
  def step(
      self,
      state: TrainState,
      batch: PyTree[Any],
      *,
      return_losses: bool = False,
      return_metrics: bool = False,
      return_summaries: bool = False,
  ) -> tuple[TrainState, Auxiliaries]:
    """Training step: forward, losses, gradients, update, and metrics."""
    # TODO(epot): Should `jax.named_call` be moved downstream directly in optax?

    # NOTE: ensure that evaluation metrics are computed from the OLD model state
    # *before* backprop gradients are applied.
    grad_fn = jax.grad(self.model_with_aux.forward, argnums=0, has_aux=True)
    grads, context = jax.named_call(grad_fn, name="grad_fn")(
        state.params,
        batch=batch,
        rngs=self.rng_streams.train_rngs(state.step),
        step=state.step,
        is_training=True,
    )
    updates, new_opt_state = jax.named_call(self.optimizer.update)(
        grads, state.opt_state, state.params
    )
    new_params = jax.named_call(optax.apply_updates)(state.params, updates)
    next_state = state.next(new_params=new_params, new_opt_state=new_opt_state)

    # add the gradients, computed updates, and *old* optimizer state to context
    context = context.replace(
        grads=grads,
        updates=updates,
        opt_state=state.opt_state,
    )

    aux = self.model_with_aux.get_aux(
        context,
        return_losses=return_losses,
        return_metrics=return_metrics,
        return_summaries=return_summaries,
    )

    return next_state, aux
