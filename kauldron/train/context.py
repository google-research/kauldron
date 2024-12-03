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

"""Context defines the namespace used for path-based querying in losses etc."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Self

import flax
from kauldron import kontext
from kauldron.train import auxiliaries

if typing.TYPE_CHECKING:
  from kauldron.train import train_step


@flax.struct.dataclass
class Context:
  """Namespace for retrieving information with path-based keys.

  The context is progressively filled during the training/eval step.

  ```python
  # Initial context contain the params, batch,...
  ctx = kd.train.Context.from_state_and_batch(state=state, batch=batch)

  # Add pred, interms, loss_states,...
  loss, ctx = model_with_aux.forward(ctx, ...)

  ```

  Attributes:
    step: The global step number. Used for evaluating schedules etc.
    batch: The input batch as returned from the data iterator.
    params: The parameters of the model. (available after the init)
    collections: Other variable collections (such as batch norm statistics).
    preds: The output of the model. (available after the model has been applied,
      e.g. for losses and metrics)
    interms: The intermediate outputs of the model as returned by
      `model.apply(..., capture_intermediates=True)`. (available after the model
      has been applied, e.g. for losses and metrics)
    loss_states: All the states of the losses as returned by
      `kd.losses.compute_losses`. (available after the forward pass)
    loss_total: The total loss value for that step (float).
    grads: The gradients of the `loss_values['total']` wrt. `params`. (available
      after the backward pass, e.g. for metrics)
    updates: The transformed gradients as returned by the optimizer. (available
      after the backward pass, e.g. for metrics)
    opt_state: The state of the optimizer prior to the update. (available after
      the backward pass, e.g. for metrics). The old state is chosen to be
      consistent with parameters which are also pre-update.
    metric_states: The states of the metrics (after the backward pass)
    summary_states: The states of the summaries (after the backward pass)
  """

  # These are always available:
  step: int
  batch: Any
  # Becomes available after model.init:
  params: Any = None
  collections: Any = None
  # Become available after model.apply:
  preds: Any = None
  interms: Any = None
  # Becomes available after loss computation:
  loss_states: Any = None
  loss_total: Any = None
  # Become available after the backward pass and optimizer:
  grads: Any = None
  updates: Any = None
  opt_state: Any = None
  # Become available after the metrics computation
  metric_states: Any = None
  summary_states: Any = None

  replace = dataclasses.replace

  @classmethod
  def from_state_and_batch(
      cls,
      *,
      state: train_step.TrainState,
      batch: Any,
  ) -> Self:
    return cls(
        step=state.step,
        params=state.params,
        collections=state.collections,
        opt_state=state.opt_state,
        batch=batch,
    )

  def flatten(self) -> dict[str, Any]:
    return kontext.flatten_with_path(self)

  def get_aux_state(
      self,
      *,
      return_losses: bool = False,
      return_metrics: bool = False,
      return_summaries: bool = False,
  ) -> auxiliaries.AuxiliariesState:
    """Returns the auxiliaries for the step."""
    return auxiliaries.AuxiliariesState(
        loss_states=self.loss_states if return_losses else None,
        metric_states=self.metric_states if return_metrics else None,
        summary_states=self.summary_states if return_summaries else None,
    )
