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

"""Flatboard utils."""

from __future__ import annotations

from typing import Sequence

from kauldron import kontext
from kauldron.train import config_lib
from kauldron.train import flatboard
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member


def _get_loss_y_keys(trainer: config_lib.Trainer) -> Sequence[str]:
  """Get a list of loss-keys for a given trainer."""
  # train losses
  loss_names = {
      k for k in kontext.flatten_with_path(trainer.train_losses, separator="/")
  }
  # evaluator losses
  for evaluator in trainer.evals.values():
    for c in evaluator.__flatboard_collections__.values():
      loss_names |= set(c.losses)

  # If more than one loss, add the total loss
  if len(loss_names) > 1:
    loss_names = {"total"} | loss_names
  return [f"losses/{l.replace('.', '/')}" for l in loss_names]


def _get_metric_y_keys(trainer: config_lib.Trainer) -> Sequence[str]:
  """Get a list of metric-keys for a given trainer."""
  metric_names = {
      k for k in kontext.flatten_with_path(trainer.train_metrics, separator="/")
  }
  # add evaluator metrics
  for evaluator in trainer.evals.values():
    for c in evaluator.__flatboard_collections__.values():
      metric_names |= set(c.metrics)
  return [f"metrics/{l.replace('.', '/')}" for l in sorted(metric_names)]


def _get_schedule_y_keys(trainer: config_lib.Trainer) -> Sequence[str]:
  """Get a list of schedule-keys for a given trainer."""
  schedule_names = [
      k for k in kontext.flatten_with_path(trainer.schedules, separator="/")
  ]
  return [f"schedules/{l.replace('.', '/')}" for l in schedule_names]


def _get_perf_stat_y_keys() -> Sequence[str]:
  """Get a list of performance statistics keys."""
  return [
      f"perf_stats/{y}"  # pylint: disable=g-complex-comprehension
      for y in [
          "steps_per_sec",
          "data_points_per_sec_global",
          "data_points_per_sec_per_device",
          "total_training_time_hours",
      ]
  ]


def _get_data_collections(trainer: config_lib.Trainer) -> list[str]:
  """Return a list of datatable collections for a given resolved trainer."""
  collections = ["train"]
  for e in trainer.evals.values():
    collections.extend(e.__flatboard_collections__.keys())
  return collections


def _get_default_dashboards(trainer: config_lib.Trainer):
  """Return the default set of Flatboard dashboards for given trainer."""
  data_collections = _get_data_collections(trainer)
  dashboards = {}
  # losses
  y_keys = _get_loss_y_keys(trainer)
  if y_keys:
    dashboards["losses"] = flatboard.DefaultDashboard(
        title="{xid}: Losses", y_keys=y_keys, collections=data_collections
    )

  # metric
  y_keys = _get_metric_y_keys(trainer)
  if y_keys:
    dashboards["metrics"] = flatboard.DefaultDashboard(
        title="{xid}: Metrics", y_keys=y_keys, collections=data_collections
    )

  # extra dashboards (e.g. for fewshot, TrainEvaluator,...)
  for eval_ in trainer.evals.values():
    dashboards.update(eval_.__flatboard_extra_dashboards__)

  # schedules
  y_keys = _get_schedule_y_keys(trainer)
  if y_keys:
    dashboards["schedules"] = flatboard.DefaultDashboard(
        title="{xid}: Schedules",
        y_keys=y_keys,
        collections=["train"],
    )

  # perf_stats
  y_keys = _get_perf_stat_y_keys()
  if y_keys:
    dashboards["perf_stats"] = flatboard.DefaultDashboard(
        title="{xid}: Performance Statistics",
        y_keys=y_keys,
        collections=["train"],
    )
  return dashboards


def add_flatboards(trainer: config_lib.Trainer):
  """Add flatboards based on trainer.flatboards or default flatboards."""
  if not status.on_xmanager or not status.is_lead_host or status.wid != 1:
    return  # only add flatboards once per experiment
  dashboard_factories = trainer.flatboards
  if not dashboard_factories:
    dashboard_factories = _get_default_dashboards(trainer)
  flatboard.add_flatboard_artifacts(dashboard_factories)
