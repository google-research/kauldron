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

"""MultiTrainStep to go along with multi_optimizer.

MultiTrainStep is a TrainStep that sequentially applies different optimizers
to different losses. It has to be used together with `multi_optimizer` to
construct the optimizers.

Note that the different gradients are available in the context.subgrads field,
and are populated after each backward pass.

Usage example:

```python
  cfg.train_losses = {
      "loss_a": ...,
      "loss_b": ...,
  }

  cfg.optimizer = kd.contrib.train.multi_optimizer(
      loss_a=optax.adam(...),
      loss_b=optax.adam(...),
  )
  cfg.train_step = kd.contrib.train.MultiTrainStep()
```
"""

import copy
import dataclasses
import functools
from typing import Any, Optional, Self

from flax import linen as nn
import flax.struct
import jax
from kauldron import kd
from kauldron.typing import PyTree  # pylint: disable=g-multiple-import,g-importing-member
import optax


@flax.struct.dataclass
class MultiContext(kd.train.Context):
  subgrads: Any = None

  @classmethod
  def from_context(cls, context: kd.train.Context, subgrads: Any) -> Self:
    """Creates a new MultiContext with the subgrads field overridden."""
    attributes = {
        f.name: getattr(context, f.name) for f in dataclasses.fields(context)
    }
    attributes["subgrads"] = subgrads
    return cls(**attributes)


@jax.named_call
def forward_with_loss(
    context: kd.train.Context | None = None,
    *,
    losses: dict[str, kd.losses.Loss],
    model: nn.Module,
    loss_name: Optional[str] = None,
    rngs: dict[str, kd.random.PRNGKey],
    is_training: bool,
    subgrad_fns: Any = None,
    **kwargs,
) -> tuple[float, MultiContext]:
  """Forward pass of the model including losses."""
  if kwargs:
    raise ValueError(
        "Further kwargs (e.g. for the legacy API) are not supported."
    )
  if subgrad_fns is None:
    subgrads = {k: context.params for k in losses}  # pytype: disable=attribute-error
  else:
    # Call all subgrad functions to get their values with gradients. The hope
    # is that XLA compilation will optimize away the repeated forward passes.
    subgrads = {
        k: subgrad_fn(context, rngs=rngs, is_training=is_training)[0].params
        for k, subgrad_fn in subgrad_fns.items()
    }
  context = MultiContext.from_context(context, subgrads)
  args, kwargs = kd.data.utils.get_model_inputs(model, context)
  preds, collections = model.apply(
      {"params": context.params} | context.collections,
      *args,
      rngs=rngs,
      mutable=True,
      capture_intermediates=True,
      is_training_property=is_training,
      **kwargs,
  )
  # Note the params can be mutable if the model call the same sub-model
  # internally but with different params. However, the updates are never
  # propagated
  collections.pop("params", None)
  interms = collections.pop("intermediates")
  context = context.replace(
      preds=preds, interms=interms, collections=collections
  )
  losses = losses if loss_name is None else {loss_name: losses[loss_name]}
  loss_total, loss_states = kd.losses.compute_losses(
      losses=losses, context=context
  )
  return loss_total, context.replace(
      loss_states=loss_states,
      loss_total=loss_total,
  )


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class MultiTrainStep(kd.train.TrainStep):
  """Multi-Optimizer Training Step."""

  def _step(
      self,
      state: kd.train.TrainState,
      batch: PyTree[Any],
  ) -> tuple[kd.train.TrainState, kd.train.Context]:
    """Training step: forward, losses, gradients, update, and metrics."""

    assert isinstance(
        self.optimizer, MultiGradientTransformation
    ), "optimizer must be a multi_optimizer"

    all_opt_states = {}
    subgrads = {}
    subgrad_fns = {}
    all_updates = {}
    all_loss_states = {}
    loss_total = 0.0
    # compute all the updates
    rngs = self.rng_streams.train_rngs(state.step)
    assert self.optimizer.loss_to_optimizer
    # Initialize context outside of the loop to silence linter.
    context = None
    for loss_name, optimizer in self.optimizer.loss_to_optimizer.items():
      context = MultiContext(
          params=state.params,
          step=state.step,
          batch=batch,
          collections=state.collections,
          subgrads=subgrads,
          opt_state=all_opt_states,
      )
      # get the loss and construct a corresponding model_with_aux
      forward = functools.partial(
          forward_with_loss,
          loss_name=loss_name,
          model=self.model,
          losses=self.aux.losses,
          subgrad_fns=copy.copy(subgrad_fns),  # copy to keep from mutating
      )
      grad_fn = jax.grad(forward, argnums=0, has_aux=True, allow_int=True)
      context_subgrad, context = jax.named_call(
          grad_fn, name=f"grad_fn_{loss_name}"
      )(context, rngs=rngs, is_training=True)
      subgrad = context_subgrad.params
      opt_state = state.opt_state[loss_name]
      updates, new_opt_state = jax.named_call(optimizer.update)(
          subgrad, opt_state, state.params
      )

      subgrads[loss_name] = subgrad
      subgrad_fns[loss_name] = grad_fn
      all_updates[loss_name] = updates
      all_opt_states[loss_name] = new_opt_state
      all_loss_states |= context.loss_states
      loss_total += context.loss_total

    # Update the parameters once after all the updates have been computed
    # that way they can hopefully all share the same forward pass
    grads = jax.tree.map(lambda *x: sum(x), *subgrads.values())
    updates = jax.tree.map(lambda *x: sum(x), *all_updates.values())
    new_params = jax.named_call(optax.apply_updates)(state.params, updates)

    next_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=all_opt_states,
        collections=context.collections,
    )

    # add the gradients, computed updates, and *old* optimizer state to context
    context = context.replace(
        subgrads=subgrads,
        grads=grads,
        updates=updates,
        opt_state=state.opt_state,
        loss_states=all_loss_states,
        loss_total=loss_total,
    )

    context = self.aux.update_context(context)
    return next_state, context


class MultiGradientTransformation(optax.GradientTransformationExtraArgs):
  loss_to_optimizer: dict[str, optax.GradientTransformation]

  def __new__(cls, init, update, loss_to_optimizer):
    mgt = super().__new__(cls, init, update)  # pylint: disable=too-many-function-args
    mgt.loss_to_optimizer = loss_to_optimizer
    return mgt


def multi_optimizer(
    **loss_to_optimizer: optax.GradientTransformation,
) -> MultiGradientTransformation:
  """Create a multi-optimizer that maps loss names to optimizers."""

  # the update function of the multi-optimizer should never be used directly
  def update(updates, state, params=None, *, loss, **extra_args):
    del updates, state, params, loss, extra_args
    raise TypeError(
        "The update function of the multi_optimizer should never be used"
        " directly. Use with MultiTrainStep instead."
    )

  # reuse the init function of the named_chain to construct a state
  init, unused_update = optax.named_chain(*loss_to_optimizer.items())

  return MultiGradientTransformation(
      init=init, update=update, loss_to_optimizer=loss_to_optimizer
  )
