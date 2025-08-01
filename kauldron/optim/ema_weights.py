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

"""Optax gradient transformation that stores an EMA version of model weights."""

import dataclasses
from typing import Any, NamedTuple, Optional

import chex
import jax.numpy as jnp
from kauldron.checkpoints import partial_loader
import optax
from optax._src import utils as optax_utils


class EmaParamsState(NamedTuple):
  """Holds an exponential moving average of model parameters."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  ema_params: optax.Params


def ema_params(
    *,
    decay: float,
    debias: bool = True,
    accumulator_dtype: Optional[Any] = None,
) -> optax.GradientTransformation:
  """Store an EMA version of model parameters.

  Different from `optax.ema`, here we do not alter the gradient. Instead, we
  maintain a copy of model parameters, which is an EMA over training steps.
  These weights can then e.g. be used during evalutation.

  NOTE: This function should be called last, e.g., at the end of `optax.chain`,
  because it applies the updates to the parameters and uses the updated
  parameters to update the EMA parameters.

  Example usage:
  ```
  cfg.optimizer = kd.optim.named_chain(**{
      "adam": optax.scale_by_adam(b1=0.95),
      "ema_params": kd.optim.ema_params(decay=0.999),
  })

  cfg.evals = {
      "ema_eval": kd.evals.Evaluator(
          init_transform=kd.optim.UseEmaParams(),
      )
  }
  ```

  Args:
    decay: Decay rate for the exponential moving average.
    debias: Whether to debias the transformed gradient.
    accumulator_dtype: Optional `dtype` to used for the accumulator; if `None`
      then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  accumulator_dtype = optax_utils.canonicalize_dtype(accumulator_dtype)

  def init_fn(params):
    return EmaParamsState(
        count=jnp.zeros([], jnp.int32),
        # Record the initial model weight.
        ema_params=optax.tree.cast(params, accumulator_dtype),
    )

  def update_fn(updates, state, params=None):
    count_inc = optax.safe_increment(state.count)
    # The model weights after optimizer step.
    new_params = optax.apply_updates(params, updates)
    # `new_ema_params = (1 - decay) * new_params + decay * old_ema_params`.
    new_ema_params = optax.update_moment(
        new_params, state.ema_params, decay, order=1
    )
    if debias:
      new_ema_params = optax.tree.bias_correction(
          new_ema_params, decay, count_inc
      )
    new_ema_params = optax.tree.cast(new_ema_params, accumulator_dtype)
    return updates, EmaParamsState(count=count_inc, ema_params=new_ema_params)

  return optax.GradientTransformation(init_fn, update_fn)


@dataclasses.dataclass(frozen=True, kw_only=True)
class UseEmaParams(partial_loader.AbstractPartialLoader):
  """Use the EMA parameters stored by the `ema_params` transform.

  Attributes:
    ema_params_transform: The index or name of the `ema_params` transform in the
      optax.chain or optax.named_chain. If not set, the last transform in the
      chain is used.
  """

  ema_params_transform: int | str | None = None

  def transform(self, state):
    """Replace the parameters with the weights from `opt_state[ema_name]`."""
    if self.ema_params_transform is None:
      # If ema_params_transform is not set, the we try to use the last transform
      # in the chain.
      if isinstance(state.opt_state, tuple):
        # This is the case for `optax.chain`.
        ema_params_state = state[-1]
      elif isinstance(state.opt_state, dict):
        # This is the case for `optax.named_chain`.
        last_key = list(state.opt_state.keys())[-1]
        ema_params_state = state.opt_state[last_key]
      else:
        raise ValueError(
            f"Unknown optimizer state type: {type(state.opt_state)}. "
            "If ema_params_transform is not set, the optimizer state needs to "
            "be a tuple or a dict (from optax.chain or optax.named_chain)."
        )
      if not hasattr(ema_params_state, "ema_params"):
        raise ValueError(
            "The last transform in the chain is not an instance of"
            " `kd.optim.ema_params`. Either make sure that `ema_params` is"
            " the last transform in the chain, or set `ema_params_transform`"
            " to the index / name of the transform."
        )
    else:
      ema_params_state = state.opt_state[self.ema_params_transform]
      if not hasattr(ema_params_state, "ema_params"):
        raise ValueError(
            "opt_state[{self.ema_params_transform}] is not an instance of"
            " `EmaParamsState`."
        )

    return state.replace(params=ema_params_state.ema_params)
