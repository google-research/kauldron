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

"""Gradient transformations."""

from __future__ import annotations
from typing import Any, Callable, NamedTuple, Optional, Union
import jax
import optax


class DecayToInitState(NamedTuple):
  """Stores the parameters at initialization."""

  init_params: optax.Params


def decay_to_init(
    weight_decay: Union[float, jax.Array],
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
) -> optax.GradientTransformation:
  """Add (params - init_params) scaled by `weight_decay`.

  This effectively acts as weight decay not towards zero but towards the
  initialization of the model. Useful for finetuning of pre-trained models.

  Args:
    weight_decay: A scalar weight decay rate.
    mask: A tree with same structure as (or a prefix of) the params PyTree, or a
      Callable that returns such a pytree given the params/updates. The leaves
      should be booleans, `True` for leaves/subtrees you want to apply the
      transformation to, and `False` for those you want to skip.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    return DecayToInitState(init_params=params)

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(
          'You are using a transformation that requires the current value of '
          'parameters, but you are not passing `params` when calling `update`.'
      )

    def _leaf_update(grad, params, init_params):
      return grad + weight_decay * (params - init_params)

    updates = jax.tree.map(
        _leaf_update, updates, params, state.init_params
    )
    # no need to update the state
    return updates, state

  # If mask is not `None`, apply mask to the gradient transformation.
  # E.g. it is common to skip weight decay on bias units and batch stats.
  if mask is not None:
    return optax.masked(optax.GradientTransformation(init_fn, update_fn), mask)
  return optax.GradientTransformation(init_fn, update_fn)
