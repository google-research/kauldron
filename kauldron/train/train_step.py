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

import flax
import flax.linen as nn
import jax
from jax.experimental import checkify
import jax.numpy as jnp
from kauldron import losses as kd_losses
from kauldron.checkpoints import checkpoint_items
from kauldron.checkpoints import partial_loader
import kauldron.data.utils as data_utils
from kauldron.train import auxiliaries
from kauldron.train import context as context_lib
from kauldron.train import rngs_lib
from kauldron.typing import ElementSpec, Float, PyTree  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import config_util
from kauldron.utils import train_property  # pylint: disable=unused-import
from kauldron.utils.sharding_utils import sharding as sharding_lib  # pylint: disable=g-importing-member
import optax

# Do not import `trainer_lib` at runtime to avoid circular imports
if typing.TYPE_CHECKING:
  from kauldron.train import trainer_lib  # pylint: disable=g-bad-import-order


_Params = PyTree[Float["..."]]
_Collections = Mapping[str, PyTree[Float["..."]]]

# Backward compatible alias as users direct import this file
AuxiliariesState = auxiliaries.AuxiliariesState


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


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class TrainStep(config_util.UpdateFromRootCfg):
  """Base Training Step.

  Subclasses can overwrite the `_step` method to implement custom training
  steps.
  """

  model: nn.Module = config_util.ROOT_CFG_REF.model
  optimizer: optax.GradientTransformation = config_util.ROOT_CFG_REF.optimizer
  rng_streams: rngs_lib.RngStreams = config_util.ROOT_CFG_REF.rng_streams
  sharding: sharding_lib.ShardingStrategy = config_util.ROOT_CFG_REF.sharding
  init_transform: partial_loader.AbstractPartialLoader = (
      config_util.ROOT_CFG_REF.init_transform
  )
  aux: auxiliaries.Auxiliaries = dataclasses.field(
      default_factory=auxiliaries.Auxiliaries
  )

  __root_cfg_fields_to_recurse__ = ("aux",)

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
    if self.optimizer is not None and not skip_optimizer:
      # Eval-only jobs do not have optimizer.
      state = self._init_optimizer(state)
    if not skip_transforms:
      # If restoring a checkpoint we can skip the (potentially slow) transforms
      state = self._init_transform(state)
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
    args, kwargs = data_utils.get_model_inputs_from_batch(self.model, batch)
    with self.sharding.set_global_mesh():
      collections = self.model.init(
          self.rng_streams.init_rngs(),
          *args,
          method=model_method,
          is_training_property=True,
          capture_intermediates=True,
          **kwargs,
      )
    collections = flax.core.unfreeze(collections)
    params = collections.pop("params", {})
    collections.pop("intermediates", None)  # Remove intermediates

    state = TrainState(  # pytype: disable=wrong-arg-types
        step=jnp.asarray(0),
        params=params,
        opt_state=None,
        collections=collections,
    )
    return sharding_lib.with_sharding_constraint(state, self.sharding.state)

  def _init_transform(self, state: TrainState) -> TrainState:
    """Run any additional init transformations and return the updated state."""
    state = self.init_transform.transform(state)
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
  ) -> tuple[TrainState, auxiliaries.AuxiliariesState]:
    """Training step: forward, losses, gradients, update, and metrics.

    Args:
      state: The training state
      batch: The batch to use for the training step
      return_losses: Whether to return the losses
      return_metrics: Whether to return the metrics
      return_summaries: Whether to return the summaries
      checkify_error_categories: Categories of errors to checkify. If empty, no
        checkify is performed.

    Returns:
      state: The updated training state
      auxiliaries: Auxiliaries containing the losses, metrics and summaries
        states.
    """
    # If reading the code, you can likely skip this function and go directly
    # to `_step`.

    # This function is just a small wrapper around `_step` for:
    # * Checkify errors handling
    # * Select which auxiliaries metrics to return.
    # * Set the output sharding
    # * Wrap the step function in the `self.sharding.set_global_mesh()` context
    #   (as some implementations of models rely on a global mesh).

    with self.sharding.set_global_mesh():
      if checkify_error_categories:
        step_fn = checkify.checkify(
            self._step, errors=checkify_error_categories
        )
        error, (state, ctx) = step_fn(state, batch)
      else:
        error = None
        state, ctx = self._step(state, batch)

    # TODO(epot): More flexible way to select the subset of context to return ?
    # And have a way to return the full context ?
    aux_state = ctx.get_aux_state(
        return_losses=return_losses,
        return_metrics=return_metrics,
        return_summaries=return_summaries,
    )
    aux_state = aux_state.replace(error=error)
    return sharding_lib.with_sharding_constraint(
        (state, aux_state),
        (self.sharding.state, self.sharding.aux),
    )

  def _step(
      self,
      state: TrainState,
      batch: PyTree[Any],
  ) -> tuple[TrainState, context_lib.Context]:
    """Training step to be wrapped by checkify and called by `step`."""
    # NOTE: ensure that evaluation metrics are computed from the OLD model state
    # *before* backprop gradients are applied.
    grad_fn = jax.grad(
        forward_with_loss,
        argnums=0,
        has_aux=True,
        allow_int=True,
    )
    # TODO(epot): Should `jax.named_call` be moved downstream directly in optax?
    grad_fn = jax.named_call(grad_fn, name="grad_fn")

    context = context_lib.Context.from_state_and_batch(state=state, batch=batch)
    context_grads, context = grad_fn(
        context,
        model=self.model,
        losses=self.aux.losses,
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

    context = self.aux.update_context(context)

    return next_state, context


def forward(
    context: context_lib.Context,
    *,
    model: nn.Module,
    rngs: rngs_lib.Rngs,
    is_training: bool,
) -> context_lib.Context:
  """Forward pass of the model.

  Arguments:
    context: Context to use for the forward pass. Should contain `params`,
      `batch`, `step`, and `collections` (and optionally `opt_state`).
    model: Model to use for the forward pass.
    rngs: Random numbers to use for the forward pass.
    is_training: Whether to run the model in training or eval mode.

  Returns:
    loss_total: Total loss.
    context: Context with the updated `loss_total`, `loss_states`,
      `interms`, and `collections`.
  """
  args, kwargs = data_utils.get_model_inputs(model, context)
  preds, collections = model.apply(
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
  return context


def forward_with_loss(
    context: context_lib.Context,
    *,
    model: nn.Module,
    losses: Mapping[str, kd_losses.Loss],
    rngs: rngs_lib.Rngs,
    is_training: bool,
) -> tuple[float, context_lib.Context]:
  """Forward pass of the model, including losses.

  Arguments:
    context: Context to use for the forward pass. Should contain `params`,
      `batch`, `step`, and `collections` (and optionally `opt_state`).
    model: Model to use for the forward pass.
    losses: Losses to compute.
    rngs: Random numbers to use for the forward pass.
    is_training: Whether to run the model in training or eval mode.

  Returns:
    loss_total: Total loss.
    context: Context with the updated `loss_total`, `loss_states`,
      `interms`, and `collections`.
  """
  context = forward(
      context=context,
      model=model,
      rngs=rngs,
      is_training=is_training,
  )
  loss_total, loss_states = kd_losses.compute_losses(
      losses=losses, context=context
  )
  return loss_total, context.replace(
      loss_states=loss_states,
      loss_total=loss_total,
  )


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class ModelWithAux(auxiliaries.Auxiliaries):
  """Model with aux.

  DEPRECATED: Do not use.
  """

  # TODO(epot): Deprecate this class in eval.

  model: nn.Module

  if typing.TYPE_CHECKING:
    # pytype fail to correctly infer the right dataclasses attributes.
    def __init__(self, **kwargs):
      pass

  def forward(self, context, **kwargs):
    return forward_with_loss(
        context=context,
        model=self.model,
        losses=self.losses,
        **kwargs,
    )

  def get_aux(self, context, **kwargs):
    return self.update_context(context).get_aux_state(**kwargs)
