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

"""Default TrainState and TrainStep implementations."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, Mapping, Optional

from etils import epy
import flax
import flax.linen as nn
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from kauldron import kontext
from kauldron import losses as kd_losses
from kauldron import metrics as kd_metrics
from kauldron import summaries as kd_summaries
from kauldron.checkpoints import checkpoint_items
from kauldron.checkpoints import partial_loader
import kauldron.data.utils as data_utils
from kauldron.train import context as context_lib
from kauldron.train import rngs_lib
from kauldron.typing import ElementSpec, Float, PyTree  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import config_util
from kauldron.utils import train_property  # pylint: disable=unused-import
from kauldron.utils.kdash import dashboard_utils
from kauldron.utils.sharding_utils import sharding as sharding_lib  # pylint: disable=g-importing-member
import optax

# Do not import `trainer_lib` at runtime to avoid circular imports
if typing.TYPE_CHECKING:
  from kauldron.train import trainer_lib  # pylint: disable=g-bad-import-order


_Params = PyTree[Float["..."]]
_Collections = Mapping[str, PyTree[Float["..."]]]


@flax.struct.dataclass
class TrainState(checkpoint_items.StandardCheckpointItem):
  """Data structure for checkpointing the model.

  Attributes:
    step: Current training step.
    params: Model parameters.
    opt_state: Optimizer state.
    collections: Mutable flax collections (e.g. `'batch_stats'`).
    training_time_hours: Training time in hours.
  """

  _: dataclasses.KW_ONLY

  step: int

  params: Optional[_Params]
  collections: Optional[_Collections]
  opt_state: Optional[PyTree[Float["..."]]]

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
  summary_states: Mapping[str, kd_metrics.State] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  # TODO(klausg): Remove `summary_kwargs` once all summaries are migrated
  summary_kwargs: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  error: checkify.Error = checkify.Error(
      _pred={}, _code={}, _metadata={}, _payload={}
  )

  def replace(self, **changes: Any) -> Auxiliaries:
    return dataclasses.replace(self, **changes)

  def merge(self, other: Optional[Auxiliaries]) -> Auxiliaries:
    """Accumulate auxiliary."""
    if other is None:
      return self
    return self.replace(
        loss_states=_reduce_states(self.loss_states, other.loss_states),
        metric_states=_reduce_states(self.metric_states, other.metric_states),
        summary_states=_reduce_states(
            self.summary_states, other.summary_states
        ),
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
    # losses
    loss_values = jax.tree.map(
        _compute_metric, self.loss_states, is_leaf=kd_metrics.State.isinstance
    )
    # Multi-process communication
    with jax.spmd_mode("allow_all"), jax.transfer_guard("allow"):
      total_loss = jnp.sum(jnp.asarray(list(loss_values.values())))

    if not isinstance(loss_values, dict):
      loss_values = dict(loss_values)  # Convert FrozenDict, ImmutableDict
    if loss_values.values():  # if there are any losses also add a total
      loss_values[dashboard_utils.TOTAL_LOSS_KEY] = total_loss

    # metrics
    metric_values = jax.tree.map(
        _compute_metric, self.metric_states, is_leaf=kd_metrics.State.isinstance
    )

    # summaries
    summary_values = jax.tree.map(
        _compute_metric,
        self.summary_states,
        is_leaf=kd_metrics.State.isinstance,
    )

    if flatten:
      metric_values = kontext.flatten_with_path(
          metric_values, prefix="metrics", separator="/"
      )
      loss_values = kontext.flatten_with_path(
          loss_values, prefix="losses", separator="/"
      )
      summary_values = kontext.flatten_with_path(
          summary_values, prefix="summaries", separator="/"
      )

    return AuxiliariesOutput(
        loss_values=loss_values,
        metric_values=metric_values,
        summary_values=summary_values,
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
  summary_values: dict[str, Any] = dataclasses.field(default_factory=dict)


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


def _gather_kwargs_with_reraise(k, summary, context):
  """Gathers summary kwargs with an error stating the offending key."""
  if not isinstance(summary, kd_summaries.Summary):
    return {}  # This is not a legacy summary, so no need to gather kwargs
  with epy.maybe_reraise(lambda: f"Error with key `{k}`: "):
    return summary.gather_kwargs(context)


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
      batch: PyTree[jax.Array],
      model_method: Optional[str] = None,
  ) -> tuple[_Params, _Collections]:
    self._assert_root_cfg_resolved()
    args, kwargs = data_utils.get_model_inputs_from_batch(self.model, batch)
    collections = self.model.init(
        init_rngs,
        *args,
        method=model_method,
        is_training_property=True,
        capture_intermediates=True,
        **kwargs,
    )
    collections = flax.core.unfreeze(collections)
    params = collections.pop("params", {})
    collections.pop("intermediates", None)  # Remove intermediates

    return params, collections

  @jax.named_call
  def forward(
      self,
      context: context_lib.Context,
      *,
      rngs: rngs_lib.Rngs,
      is_training: bool,
  ) -> tuple[float, context_lib.Context]:
    """Forward pass of the model including losses.

    Arguments:
      context: Context to use for the forward pass. Should contain `params`,
        `batch`, `step`, and `collections` (and optionally `opt_state`).
      rngs: Random numbers to use for the forward pass.
      is_training: Whether to run the model in training or eval mode.

    Returns:
      loss_total: Total loss.
      context: Context with the updated `loss_total`, `loss_states`,
        `interms`, and `collections`.
    """
    args, kwargs = data_utils.get_model_inputs(self.model, context)
    preds, collections = self.model.apply(
        {"params": context.params} | context.collections,
        *args,
        rngs=rngs,
        mutable=True,
        capture_intermediates=True,  # TODO(klausg): check if need a filter here
        is_training_property=is_training,
        **kwargs,
    )
    # Note the params can be mutable if the model call the same sub-model
    # internally but with different params. However, the updates are never
    # propagated
    collections.pop("params", None)
    interms = collections.pop("intermediates")
    context = context.replace(
        preds=preds,
        interms=interms,
        collections=collections,
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
          metric_states=jax.tree.map(
              lambda m: m.get_state_from_context(context), self.metrics
          )
      )

    if return_summaries:
      # TODO(klausg): remove legacy summaries protocol once all are migrated
      # legacy summaries protocol:
      aux = aux.replace(
          summary_kwargs={
              k: _gather_kwargs_with_reraise(k, summary, context)
              for k, summary in self.summaries.items()
          }
      )
      # new summaries as metrics protocol:
      def _get_summary_state(summary):
        if isinstance(summary, kd_metrics.Metric):
          return summary.get_state_from_context(context)
        else:
          return kd_metrics.EmptyState()

      aux = aux.replace(
          summary_states=jax.tree.map(_get_summary_state, self.summaries)
      )
    return aux


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class TrainStep(config_util.UpdateFromRootCfg):
  """Training Step."""

  model_with_aux: ModelWithAux = dataclasses.field(default_factory=ModelWithAux)
  optimizer: optax.GradientTransformation = config_util.ROOT_CFG_REF.optimizer
  rng_streams: rngs_lib.RngStreams = config_util.ROOT_CFG_REF.rng_streams
  sharding: sharding_lib.ShardingStrategy = config_util.ROOT_CFG_REF.sharding
  init_transforms: Mapping[str, partial_loader.AbstractPartialLoader] = (
      config_util.ROOT_CFG_REF.init_transforms
  )

  __root_cfg_fields_to_recurse__ = ("model_with_aux",)

  def init(
      self,
      elem_spec: ElementSpec,
      *,
      model_method: Optional[str] = None,
      skip_transforms: bool = False,
      skip_optimizer: bool = False,
  ) -> TrainState:
    """Initialize the model and return the initial TrainState.

    Args:
      elem_spec: Structure of the input batch
      model_method: Name of the flax model method (default to `__call__`)
      skip_transforms: If `False`, apply the `init_transform` on the state (e.g.
        to overwrite the weights with ones from another checkpoint).
      skip_optimizer: If `True`, do not initialize the optimizer.

    Returns:
      state: The training state
    """
    self._assert_root_cfg_resolved()
    if isinstance(elem_spec, dict):
      elem_spec = flax.core.freeze(elem_spec)
    state = self._init_model(elem_spec, model_method=model_method)
    if not skip_transforms:
      # If restoring a checkpoint we can skip the (potentially slow) transforms
      state = self._init_transforms(state)
    if self.optimizer is not None and not skip_optimizer:
      # Eval-only jobs do not have optimizer.
      state = self._init_optimizer(state)
    return state

  @functools.partial(
      jax.jit,
      static_argnames=("self", "elem_spec", "model_method"),
  )
  def _init_model(
      self,
      elem_spec: ElementSpec,
      *,
      model_method: Optional[str] = None,
  ) -> TrainState:
    """Initialize the model and return the initial TrainState."""
    batch = data_utils.mock_batch_from_elem_spec(elem_spec, self.sharding.ds)
    params, collections = self.model_with_aux.init(
        self.rng_streams.init_rngs(),
        batch,
        model_method=model_method,
    )
    state = TrainState(  # pytype: disable=wrong-arg-types
        step=jnp.asarray(0),
        params=params,
        opt_state=None,
        collections=collections,
    )
    return sharding_lib.with_sharding_constraint(state, self.sharding.state)

  def _init_transforms(self, state: TrainState) -> TrainState:
    """Run any additional init transformations and return the updated state."""
    for init_transf in self.init_transforms.values():
      state = init_transf.transform(state)
    # Transforms should ideally propagate the sharding from the state, but in
    # case they forget, we explicitly re-apply the sharding.
    return sharding_lib.with_sharding_constraint(state, self.sharding.state)

  @functools.partial(
      jax.jit,
      static_argnames="self",
      donate_argnames=("state",),
  )
  def _init_optimizer(
      self,
      state: TrainState,
  ) -> TrainState:
    """Initialize the model and return the initial TrainState."""
    opt_state = self.optimizer.init(state.params)
    state = state.replace(opt_state=opt_state)
    return sharding_lib.with_sharding_constraint(state, self.sharding.state)

  @functools.partial(
      jax.jit,
      static_argnames=(
          "self",
          "return_losses",
          "return_metrics",
          "return_summaries",
          "checkify_error_categories",
      ),
      donate_argnames=("state",),
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
      checkify_error_categories: frozenset[
          trainer_lib.CheckifyErrorCategory
      ] = frozenset(),
  ) -> tuple[TrainState, Auxiliaries]:
    """Training step: forward, losses, gradients, update, and metrics."""
    if checkify_error_categories:
      step_fn = checkify.checkify(self._step, errors=checkify_error_categories)
      error, (state, aux) = step_fn(
          state,
          batch,
          return_losses=return_losses,
          return_metrics=return_metrics,
          return_summaries=return_summaries,
      )
      aux = aux.replace(error=error)
    else:
      state, aux = self._step(
          state,
          batch,
          return_losses=return_losses,
          return_metrics=return_metrics,
          return_summaries=return_summaries,
      )

    return state, aux

  def _step(
      self,
      state: TrainState,
      batch: PyTree[Any],
      *,
      return_losses: bool = False,
      return_metrics: bool = False,
      return_summaries: bool = False
  ) -> tuple[TrainState, Auxiliaries]:
    """Training step to be wrapped by checkify and called by `step`."""
    # TODO(epot): Should `jax.named_call` be moved downstream directly in optax?
    # NOTE: ensure that evaluation metrics are computed from the OLD model state
    # *before* backprop gradients are applied.
    grad_fn = jax.grad(
        self.model_with_aux.forward,
        argnums=0,
        has_aux=True,
        allow_int=True,
    )
    grad_fn = jax.named_call(grad_fn, name="grad_fn")

    context = context_lib.Context.from_state_and_batch(state=state, batch=batch)
    context_grads, context = grad_fn(
        context,
        rngs=self.rng_streams.train_rngs(state.step),
        is_training=True,
    )
    params_grads = context_grads.params
    assert isinstance(context, context_lib.Context)
    updates, new_opt_state = jax.named_call(self.optimizer.update)(
        params_grads, state.opt_state, state.params
    )
    new_params = jax.named_call(optax.apply_updates)(state.params, updates)

    next_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        collections=context.collections,
    )

    # add the gradients, computed updates, and *old* optimizer state to context
    context = context.replace(
        grads=params_grads,
        updates=updates,
        opt_state=state.opt_state,
    )

    aux = self.model_with_aux.get_aux(
        context,
        return_losses=return_losses,
        return_metrics=return_metrics,
        return_summaries=return_summaries,
    )

    return sharding_lib.with_sharding_constraint(
        (next_state, aux),
        (self.sharding.state, self.sharding.aux),
    )
