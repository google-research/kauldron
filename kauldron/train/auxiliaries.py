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

"""Auxiliaries."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

import flax
import jax
from jax.experimental import checkify
from kauldron import kontext
from kauldron import losses as kd_losses
from kauldron import metrics as kd_metrics
from kauldron.train import context as context_lib
from kauldron.utils import config_util
from kauldron.utils import immutabledict
from kauldron.utils.kdash import dashboard_utils
import numpy as np


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class Auxiliaries(config_util.UpdateFromRootCfg):
  """Wrapper around the losses, summaries and metrics."""

  losses: Mapping[str, kd_losses.Loss] = config_util.ROOT_CFG_REF.train_losses
  metrics: Mapping[str, kd_metrics.Metric] = (
      config_util.ROOT_CFG_REF.train_metrics
  )
  summaries: Mapping[str, kd_metrics.Metric] = (
      config_util.ROOT_CFG_REF.train_summaries
  )

  def __post_init__(self):
    immutabledict.freeze_dict_attrs(self, ["losses", "metrics", "summaries"])

  @jax.named_call
  def update_context(self, context: context_lib.Context) -> context_lib.Context:
    """Get auxilaries."""

    # TODO(epot): Cleanup loss-states:
    # * Split `kd/losses/base:compute_losses` into `get_state` and
    #   `compute_losses(loss_states) -> float`
    # * Unify all the `m.get_state_from_context` patterns for metrics,
    #   summaries, and losses.

    # Compute the loss states here if missing (e.g. in eval or when
    # `kd.train.forward` is called rather than `kd.train.forward_with_loss`)
    if context.loss_states is None:
      loss_states = jax.tree.map(
          lambda m: m.get_state_from_context(context), self.losses
      )
    else:
      loss_states = context.loss_states

    metric_states = jax.tree.map(
        lambda m: m.get_state_from_context(context), self.metrics
    )
    summary_states = jax.tree.map(
        lambda m: m.get_state_from_context(context), self.summaries
    )

    return dataclasses.replace(
        context,
        loss_states=loss_states,
        metric_states=metric_states,
        summary_states=summary_states,
    )


@flax.struct.dataclass
class AuxiliariesState:
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
  error: checkify.Error = checkify.Error(
      _pred={}, _code={}, _metadata={}, _payload={}
  )

  def replace(self, **changes: Any) -> AuxiliariesState:
    return dataclasses.replace(self, **changes)

  def merge(self, other: Optional[AuxiliariesState]) -> AuxiliariesState:
    """Accumulate auxiliary."""
    # TODO(epot): Remove merging with `None`. Instead, can merge with an empty
    # `AuxiliariesState()`.
    if other is None:
      return self
    return self.replace(
        loss_states=_reduce_states(self.loss_states, other.loss_states),
        metric_states=_reduce_states(self.metric_states, other.metric_states),
        summary_states=_reduce_states(
            self.summary_states, other.summary_states
        ),
    )

  def finalize(self) -> AuxiliariesState:
    """Finalizes the auxiliary state."""
    with jax.transfer_guard("allow"):
      return self.replace(
          loss_states=_finalize_states(self.loss_states),
          metric_states=_finalize_states(self.metric_states),
          summary_states=_finalize_states(self.summary_states),
      )

  def __or__(self, other: AuxiliariesState | None) -> AuxiliariesState:
    """Alias for `.merge()`: `aux = aux1 | aux2`."""
    if other is None:
      return self
    return self.merge(other)

  def __ror__(self, other: AuxiliariesState | None) -> AuxiliariesState:
    """Alias for `.merge()`: `aux = aux1 | aux2`."""
    if other is None:
      return self
    return other.merge(self)

  def compute(self, *, flatten: bool = True) -> AuxiliariesOutput:
    """Compute losses and metrics."""
    final = self.finalize()
    # losses
    loss_values = jax.tree.map(
        _compute_metric, final.loss_states, is_leaf=kd_metrics.State.isinstance
    )

    if not isinstance(loss_values, dict):
      # Convert FrozenDict, ImmutableDict
      loss_values = _nested_mappings_to_dict(loss_values)
    if loss_values.values():  # if there are any losses also add a total
      total_loss = jax.tree.reduce(lambda x, y: x + y, loss_values)
      loss_values[dashboard_utils.TOTAL_LOSS_KEY] = total_loss

    # metrics
    metric_values = jax.tree.map(
        _compute_metric,
        final.metric_states,
        is_leaf=kd_metrics.State.isinstance,
    )

    # summaries
    summary_values = jax.tree.map(
        _compute_metric,
        final.summary_states,
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


@flax.struct.dataclass
class AuxiliariesOutput:
  """Auxiliaries final values (after merge and compute)."""

  loss_values: dict[str, Any] = dataclasses.field(default_factory=dict)
  metric_values: dict[str, Any] = dataclasses.field(default_factory=dict)
  summary_values: dict[str, Any] = dataclasses.field(default_factory=dict)


def _convert_leaf(leaf):
  if isinstance(leaf, jax.Array):
    return np.asarray(leaf)
  return leaf


def _nested_mappings_to_dict(obj):
  """Recursively converts nested mappings to standard Python dictionaries."""
  if isinstance(obj, Mapping):
    return {k: _nested_mappings_to_dict(v) for k, v in obj.items()}
  else:
    return obj


def _compute_metric(state: Any):
  """Compute the value of a metric for a given state and return the result."""
  # Accept cross-process computation (some metrics cannot be jitted)
  with jax.transfer_guard("allow"):
    result = state.compute()
    # Convert all results from jax.Array to np.array.
    # We do this to ensure that the metric writers do not accidentally invoke
    # any jax operations. This is important because metric writers are often
    # multi-threaded, and in a multi-host setup this can lead to problems.
    # See cl/751110291 for more context.  # copybara: strip
    return jax.tree.map(_convert_leaf, result)


def _reduce_states_single(*states: kd_metrics.State) -> kd_metrics.State:
  final_state, *rest_states = states
  for state in rest_states:
    final_state = final_state.merge(state)
  return final_state


def _reduce_states(
    *all_states: Mapping[str, kd_metrics.State]
) -> dict[str, kd_metrics.State]:
  """Merge all the states from the different metrics."""
  # Filter empty states (e.g. created by empty `kd.train.Auxiliaries()`)
  all_states = tuple(state for state in all_states if state)
  if not all_states:
    return {}
  return jax.tree.map(
      _reduce_states_single,
      *all_states,
      is_leaf=lambda x: isinstance(x, kd_metrics.State),
  )


def _finalize_states(
    states: Mapping[str, kd_metrics.State],
) -> dict[str, kd_metrics.State]:
  """finalize all the states from the different metrics."""
  return jax.tree.map(
      lambda x: x.finalize() if isinstance(x, kd_metrics.State) else x,
      states,
      is_leaf=lambda x: isinstance(x, kd_metrics.State),
  )
