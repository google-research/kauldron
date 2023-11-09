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

"""Context defines the namespace used for path-based querying in losses etc."""

from __future__ import annotations

import dataclasses
from typing import Any
import flax
from kauldron.utils import paths


@flax.struct.dataclass
class Context:
  """Namespace for retrieving information with path-based keys.

  Attributes:
    step: The global step number. Used for evaluating schedules etc.
    batch: The input batch as returned from the data iterator.
    params: The parameters of the model. (available after the init)
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
  """

  # These are always available:
  step: int
  batch: Any
  # Becomes available after model.init:
  params: Any = None
  # Become available after model.apply:
  preds: Any = None
  interms: Any = None
  # Becomes available after loss computation:
  loss_states: Any = None
  loss_total: Any = None
  # Become available after the backward pass and optimizer:
  grads: Any = None
  updates: Any = None

  replace = dataclasses.replace

  def flatten(self) -> dict[str, Any]:
    return paths.flatten_with_path(self)
