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
import jax
import jax.numpy as jnp
from kauldron import kontext
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

    # Debias logic taken from SD:
    # https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/ema.py#L30
    if debias:
      debiased_decay = jnp.minimum(decay, (1 + count_inc) / (10 + count_inc))
    else:
      debiased_decay = decay

    # The model weights after optimizer step.
    new_params = optax.apply_updates(params, updates)
    # `new_ema_params = (1 - decay) * new_params + decay * old_ema_params`.
    new_ema_params = optax.update_moment(
        new_params, state.ema_params, debiased_decay, order=1
    )
    new_ema_params = optax.tree.cast(new_ema_params, accumulator_dtype)
    return updates, EmaParamsState(count=count_inc, ema_params=new_ema_params)

  return optax.GradientTransformation(init_fn, update_fn)


@dataclasses.dataclass(frozen=True, kw_only=True)
class UseEmaParams(partial_loader.AbstractPartialLoader):
  """Use the EMA parameters stored by the `ema_params` transform.

  Attributes:
    ema_params_transform: The path to the `ema_params` transform in the
      optax.chain or optax.named_chain. If not set, the state is searched for an
      `EmaParamsState` (which has to be unique).
    partial_ok: If `True`, missing EMA params are ignored and the original
      params are kept. This is useful in combination with frozen parameters,
      e.g. when using with `kd.optim.partial_updates`.
  """

  ema_params_transform: str | None = None

  partial_ok: bool = False

  def transform(self, state):
    """Replace the parameters with the params from the EMA state."""
    eparams = self._get_ema_params(state.opt_state)

    def _merge_params(path, params, other):
      if other is None or isinstance(other, optax.MaskedNode):
        if not self.partial_ok:
          path = kontext.Path.from_jax_path(path)
          raise KeyError(
              f"No EMA params found for path {path}. Set `partial_ok=True` to"
              " allow missing EMA params."
          )
        return params
      else:
        return other

    updated_params = jax.tree.map_with_path(
        _merge_params, state.params, eparams
    )

    return state.replace(params=updated_params)

  def _get_ema_params(self, state):
    ema_params_path = self.ema_params_transform
    if ema_params_path is None:
      # To find the path of the EmaParamsState we first replace all instances
      # of EmaParamsState with its path and all other leaves with None.
      state_to_path = jax.tree.map_with_path(
          _replace_ema_params_state_with_path,
          state,
          is_leaf=lambda x: isinstance(x, EmaParamsState),
      )
      # Then we aggregate all paths into a single set via jax.tree.reduce.
      possible_paths = jax.tree.reduce(
          lambda x, y: x | {y}, state_to_path, set()
      )
      if len(possible_paths) > 1:
        raise ValueError(
            f"Found multiple EmaParamsStates ({possible_paths}). Please"
            " set `ema_params_transform` to the path of the desired"
            " EmaParamsState manually."
        )
      if not possible_paths:
        raise ValueError(
            "No EmaParamsState found. Please set `ema_params_transform`"
            " to the path of the EmaParamsState manually."
        )
      ema_params_path = possible_paths.pop()

    ema_params_state = kontext.get_by_path(state, ema_params_path)
    if not hasattr(ema_params_state, "ema_params"):
      raise ValueError(
          f"The object at state.'{ema_params_path}' is not an instance of"
          " `EmaParamsState` (it is missing an 'ema_params' attribute)."
      )
    return ema_params_state.ema_params


def _replace_ema_params_state_with_path(p, x):
  if isinstance(x, EmaParamsState):
    return kontext.Path.from_jax_path(p)
  else:
    return None
