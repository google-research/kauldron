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

"""Flatboard dashboard structure."""

from __future__ import annotations

import abc
from collections.abc import Iterable
import copy
import dataclasses

from etils import epy
from kauldron.losses import base as base_loss
from kauldron.metrics import base
from kauldron.utils.kdash import plot_utils

TOTAL_LOSS_KEY = 'total'


class DashboardsBase(abc.ABC):
  """Flatboard dashboard structure.

  Flatboard dashboards have the following structure:

  * Dashboard (e.g. `metrics`, `schedules`)
    * Plot (e.g. `Accuracy`, `CrossEntropy`,...)
      * Query (i.e. collections, like `train`, `eval`)

  However for convenience, it's sometimes easier to define the structure the
  other way around (collection -> plot -> metric, or collection -> metric ->
  plot).
  The `DashboardsBase` abstraction allow to define partial dashboards in
  arbitrary ways and merge them later on.

  There's 4 main dashboard abstractions:

  * SingleDashboard: Simple dashboard (with multiple plots).
  * MultiDashboards: Container of multiple dashboards.
  * MetricDashboards: Standard dashboards for the `metrics`, `losses`.
  * NoopDashboard: Empty dashboard.

  Usage:

  ```python
  dashboards = kdash.MultiDashboards.from_iterable([
      kdash.MetricDashboards(
          collection='train',
          losses={'xent': None},
          metrics={'accuracy': TopKAccuracy, 'psnr': None},
      ),
      kdash.MetricDashboards(
          collection='eval',
          losses={'xent': None},
          metrics={
            'accuracy': TopKAccuracy,
            'precision_recall': PrecisionRecallMetric
          },
      ),
  ])

  kdash.build_and_upload(dashboards)
  ```
  """

  @abc.abstractmethod
  def normalize(self) -> MultiDashboards:
    """Returns a normalized version of the dashboard."""
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True, kw_only=True)
class SingleDashboard(DashboardsBase):
  """Single dashboard containing multiple plots.

  Attributes:
    name: Name of the dashboard (used for XManager artifact, XM UI plot tabs).
      For example: `metrics`, `losses`, `perf_stats`...
    title: Title of the dashboard. This can contain {xid} placeholders. For
      example: `{xid}: Performance statistics`
    plots: List of plots to display in the dashboard.
  """

  name: str
  title: str
  plots: list[plot_utils.Plot]

  @classmethod
  def from_y_keys(
      cls,
      *,
      name: str,
      title: str,
      y_keys: list[str],
      collections: list[str],  # pylint: disable=redefined-outer-name
  ) -> SingleDashboard:
    return cls(
        name=name,
        title=title,
        plots=[
            plot_utils.Plot(y_key=y_key, collections=collections)
            for y_key in y_keys
        ],
    )

  def normalize(self) -> MultiDashboards:
    return MultiDashboards(dashboards={self.name: self})

  def build(self, ctx: plot_utils.BuildContext) -> fb.Dashboard:
    title = self.title.format(
        xid=ctx.id,
    )
    return fb.Dashboard(
        title=title,
        plots=[plot.build(ctx) for plot in self.plots],
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoopDashboard(DashboardsBase):
  """Empty dashboard."""

  def normalize(self) -> MultiDashboards:
    return MultiDashboards(dashboards={})


@dataclasses.dataclass(frozen=True, kw_only=True)
class MetricDashboards(DashboardsBase):
  """Standard `metrics` & `losses` dashboards for a single collection.

  All `MetricDashboards` from the various collections are merged together.
  """
  collection: str
  losses: dict[str, base_loss.Loss | None] = dataclasses.field(
      default_factory=dict
  )
  metrics: dict[str, base.Metric | None] = dataclasses.field(
      default_factory=dict
  )

  def __post_init__(self):
    object.__setattr__(self, 'losses', copy.copy(self.losses))
    object.__setattr__(self, 'metrics', copy.copy(self.metrics))

  def normalize(self) -> MultiDashboards:
    losses = dict(self.losses)

    # If more than one loss, add the total loss. Not the total loss is
    # displayed per-collection.
    if len(losses) > 1:
      losses[TOTAL_LOSS_KEY] = None

    return MultiDashboards.from_iterable([
        SingleDashboard.from_y_keys(
            name='losses',
            title='{xid}: Losses',
            y_keys=_get_key(losses, prefix='losses'),
            collections=[self.collection],
        ),
        SingleDashboard.from_y_keys(
            name='metrics',
            title='{xid}: Metrics',
            y_keys=_get_key(self.metrics, prefix='metrics'),
            collections=[self.collection],
        ),
    ])


@dataclasses.dataclass(frozen=True, kw_only=True)
class MultiDashboards(DashboardsBase):
  """Container of multiple dashboards."""
  dashboards: dict[str, SingleDashboard]

  @classmethod
  def from_iterable(
      cls, dashboards: Iterable[DashboardsBase]
  ) -> MultiDashboards:
    """Factory from an iterable of dashboards."""
    merged_dashboards = {}
    for dashboard in dashboards:
      for name, dash in dashboard.normalize().dashboards.items():
        if name not in merged_dashboards:  # New dashboard, add it.
          merged_dashboards[name] = dash
        else:  # Existing dashboard, merge plots.
          prev_dash = merged_dashboards[name]
          # Validation:
          # TODO(epot): Generic validation which check all fields except a list
          if prev_dash.title != dash.title:
            raise ValueError(
                f'Dashboard title mismatch: {prev_dash.title} != {dash.title}\n'
                f'* Dashboard 1: {prev_dash}\n'
                f'* Dashboard 2: {dash}'
            )
          merged_dashboards[name] = dataclasses.replace(
              prev_dash, plots=_merge_plots(prev_dash.plots + dash.plots)
          )
    return cls(dashboards=merged_dashboards)

  def normalize(self) -> MultiDashboards:
    return self

  def build(self, ctx: plot_utils.BuildContext) -> dict[str, fb.Dashboard]:
    return {
        name: dash.build(ctx)
        for name, dash in self.dashboards.items()
        # Filter empty dashboards (should we do this here ?)
        if dash.plots
    }


def _merge_plots(plots: list[plot_utils.Plot]) -> list[plot_utils.Plot]:
  """Merges plots with the same y_key."""

  plots_by_key = epy.groupby(plots, key=lambda p: (p.x_key, p.y_key))
  merged_plots = []
  for plot_for_key in plots_by_key.values():
    new_plot = plot_utils.Plot.merge(plot_for_key)
    merged_plots.append(new_plot)
  return merged_plots


def _get_key(
    metrics: dict[str, base.Metric | None], prefix: str = 'metrics'
) -> list[str]:
  """Get metrics' keys."""
  y_keys = []
  for name, sub_names in metrics.items():
    if isinstance(sub_names, base.Metric):
      sub_names = sub_names.__metric_names__()

    if sub_names is None:
      y_keys.append(f'{prefix}/{name}')
    else:
      for sub_name in sub_names:
        y_keys.append(f'{prefix}/{name}/{sub_name}')
  return y_keys
