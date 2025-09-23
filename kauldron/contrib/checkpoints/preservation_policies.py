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

"""Custom preservation policies for Orbax checkpoint managers."""

import dataclasses
from typing import Sequence
from absl import logging
from kauldron import kontext
import numpy as np
import orbax.checkpoint as ocp

PreservationPolicy = ocp.checkpoint_managers.PreservationPolicy
PreservationContext = ocp.checkpoint_managers.PreservationContext
PolicyCheckpointInfo = ocp.checkpoint_managers.PolicyCheckpointInfo


@dataclasses.dataclass
class ExpStep(PreservationPolicy):
  """Keep exponentially spaced checkpoints.

  E.g. for interval=100 and base=2, the following checkpoints are preserved:
  [0, 100, 200, 400, 800, 1600, ...]

  Attributes:
    interval: The minimum interval at which checkpoints are preserved.
    base: The base of the exponential schedule. E.g. for base=2, the interval
      doubles after each checkpoint.
  """

  interval: int = 100
  base: int = 2

  def should_preserve(
      self,
      checkpoints: Sequence[PolicyCheckpointInfo],
      *,
      context: PreservationContext,
  ) -> Sequence[bool]:
    if self.interval == 0:
      raise ValueError("min_interval must not be 0.")
    steps = np.array([info.step / self.interval for info in checkpoints])
    logstep = np.log2(steps, where=steps > 0) / np.log2(self.base)
    result = logstep % 1.0 == 0
    _log_preservation_decision(
        f"ExpStep (min_interval={self.interval}, base={self.base})",
        checkpoints,
        result,
    )
    return result


def keep_best_n(
    metric_path: str,
    n: int | None,
    reverse: bool = False,
    keep_checkpoints_without_metrics: bool = True,
) -> ocp.checkpoint_managers.BestN:
  """Returns a preservation policy that keeps the best n checkpoints.

  Args:
    metric_path: Path to the metric to use for comparison. E.g.
      "my_eval.accuracy".
    n: The number of best checkpoints to keep (top-n). If set to None, all
      checkpoints are preserved. If 0, no checkpoints are preserved.
    reverse: Set to True to keep the n checkpoints with the lowest metric value.
    keep_checkpoints_without_metrics: Whether to keep checkpoints without
      metrics.

  Returns:
    A BestN preservation policy.
  """

  def _get_metric_fn(metrics) -> float:
    return float(kontext.get_by_path(metrics, metric_path))

  return ocp.checkpoint_managers.BestN(
      n=n,
      get_metric_fn=_get_metric_fn,
      reverse=reverse,
      keep_checkpoints_without_metrics=keep_checkpoints_without_metrics,
  )


def keep_exp_step_and_best_n(
    interval: int,
    metric_path: str,
    n: int | None,
    base: int = 2,
    reverse: bool = False,
    keep_checkpoints_without_metrics: bool = True,
) -> PreservationPolicy:
  """Preserves the n best as well as exponentially spaced checkpoints.

  Args:
    interval: The minimum interval at which checkpoints are preserved.
    metric_path: Path to the metric to use for comparison. E.g.
      "my_eval.accuracy".
    n: The number of best checkpoints to keep (top-n). If set to None, all
      checkpoints are preserved. If 0, no checkpoints are preserved.
    base: The base of the exponential schedule. E.g. for base=2, the interval
      doubles after each checkpoint.
    reverse: Set to True to keep the n checkpoints with the lowest metric value.
    keep_checkpoints_without_metrics: Whether to keep checkpoints without
      metrics.

  Returns:
    An AnyPreservationPolicy that preserves exponentially spaced checkpoints
    and the n best checkpoints.
  """
  return ocp.checkpoint_managers.AnyPreservationPolicy(
      policies=[
          ExpStep(interval=interval, base=base),
          keep_best_n(
              metric_path, n, reverse, keep_checkpoints_without_metrics
          ),
      ]
  )


def _log_preservation_decision(
    policy_name: str,
    checkpoints: Sequence[PolicyCheckpointInfo],
    should_preserve_list: Sequence[bool],
):
  """Logs preservation decisions."""
  if logging.vlog_is_on(1):
    for i, checkpoint in enumerate(checkpoints):
      if should_preserve_list[i]:
        logging.vlog(
            1,
            f" {policy_name}: Preserving checkpoint at step"
            f" {checkpoint.step}).",
        )
      else:
        logging.vlog(
            1,
            f" {policy_name}: Not preserving checkpoint at step"
            f" {checkpoint.step}).",
        )
