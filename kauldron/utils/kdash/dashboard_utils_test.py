# Copyright 2026 The kauldron Authors.
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

"""Test metrics logging."""

from __future__ import annotations

import dataclasses

import flax.struct
from kauldron import kd
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.utils.kdash import dashboard_utils
from kauldron.utils.kdash import plot_utils
import pytest


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class IntAverage(base.Metric):
  x: kontext.Key = 'batch.x'
  mask: kontext.Key = None

  def __metric_names__(self):
    return ['average', 'twice_average']

  @flax.struct.dataclass
  class State(base_state.AverageState):  # pyrefly: ignore[bad-override]

    def compute(self):
      return int(super().compute())

  def get_state(self, x=None) -> IntAverage.State:  # pyrefly: ignore[bad-override]
    return dict(  # pyrefly: ignore[bad-return]
        average=IntAverage.State.from_values(values=x),
        twice_average=IntAverage.State.from_values(values=2*x)  # pyrefly: ignore[unsupported-operation]
    )


def test_get_key():
  # Test when input = {}
  out = dashboard_utils._get_key({}, 'losses')
  assert not out

  # Test when input = {xx: None}
  out = dashboard_utils._get_key({'Average': None}, 'metrics')
  assert out == ['metrics/Average']

  out = dashboard_utils._get_key({'Average': None}, 'losses')
  assert out == ['losses/Average']

  # Test when input = {xx: Metrics}
  m = IntAverage()
  out = dashboard_utils._get_key({'Average': m}, 'metrics')
  expected_out = [f'metrics/Average/{name}' for name in m.__metric_names__()]
  assert out == expected_out


def test_overview_dashboard_custom_and_opt_out():
  custom_overview = dashboard_utils.SingleDashboard(
      name='overview',
      title='Custom Overview',
      plots=[plot_utils.Plot(y_key='custom_metric', collections=['train'])],
  )
  opt_out_db = dashboard_utils.SingleDashboard(
      name='opt_out_db',
      title='Opt Out DB',
      plots=[plot_utils.Plot(y_key='opt_out_metric', collections=['train'])],
      in_overview=False,
  )
  normal_db = dashboard_utils.SingleDashboard(
      name='normal_db',
      title='Normal DB',
      plots=[plot_utils.Plot(y_key='normal_metric', collections=['train'])],
  )
  # Test custom overview is respected and moved first
  multi = dashboard_utils.MultiDashboards.from_iterable([
      normal_db,
      custom_overview,
      opt_out_db,
  ])
  res = multi.add_overview_dashboard()
  assert list(res.dashboards.keys()) == ['overview', 'normal_db', 'opt_out_db']
  assert res.dashboards['overview'] == custom_overview

  # Test in_overview=False is respected when generating overview
  multi2 = dashboard_utils.MultiDashboards.from_iterable([
      normal_db,
      opt_out_db,
  ])
  res2 = multi2.add_overview_dashboard()
  assert list(res2.dashboards.keys()) == ['overview', 'normal_db', 'opt_out_db']
  assert res2.dashboards['overview'].plots == [
      plot_utils.Plot(
          y_key='normal_metric',
          collections=['train'],
          remove_prefix=False,
      )
  ]


def test_metric_dashboards_opt_out():
  metric_db = dashboard_utils.MetricDashboards(
      collection='train',
      losses={'xent': None},
      metrics={'acc': None},
      in_overview=False,
  )
  res = metric_db.normalize().add_overview_dashboard()
  assert 'overview' not in res.dashboards


def test_overview_dashboard_disabled():
  test_db = dashboard_utils.SingleDashboard(
      name='test_db',
      title='Test DB',
      plots=[plot_utils.Plot(y_key='test_metric', collections=['train'])],
  )
  multi = dashboard_utils.MultiDashboards.from_iterable(
      [test_db], create_overview_dashboard=False
  )
  res = multi.add_overview_dashboard()
  assert 'overview' not in res.dashboards
  assert list(res.dashboards.keys()) == ['test_db']


def test_multi_dashboards_merge_in_overview_mismatch():
  db1 = dashboard_utils.SingleDashboard(
      name='metrics',
      title='Metrics',
      plots=[plot_utils.Plot(y_key='metric1', collections=['train'])],
      in_overview=True,
  )
  db2 = dashboard_utils.SingleDashboard(
      name='metrics',
      title='Metrics',
      plots=[plot_utils.Plot(y_key='metric2', collections=['eval'])],
      in_overview=False,
  )
  with pytest.raises(ValueError, match='in_overview mismatch'):
    dashboard_utils.MultiDashboards.from_iterable([db1, db2])


def test_multi_dashboards_merge_in_overview_both_false():
  db1 = dashboard_utils.SingleDashboard(
      name='metrics',
      title='Metrics',
      plots=[plot_utils.Plot(y_key='metric1', collections=['train'])],
      in_overview=False,
  )
  db2 = dashboard_utils.SingleDashboard(
      name='metrics',
      title='Metrics',
      plots=[plot_utils.Plot(y_key='metric2', collections=['eval'])],
      in_overview=False,
  )
  merged = dashboard_utils.MultiDashboards.from_iterable([db1, db2])
  assert not merged.dashboards['metrics'].in_overview
  res = merged.add_overview_dashboard()
  assert 'overview' not in res.dashboards


@dataclasses.dataclass(frozen=True, kw_only=True)
class CustomDashboardNoInOverview(dashboard_utils.DashboardsBase):
  name: str
  title: str
  plots: list[plot_utils.Plot]

  def normalize(self) -> dashboard_utils.MultiDashboards:
    return dashboard_utils.MultiDashboards(
        dashboards={self.name: self}  # pyrefly: ignore[bad-argument-type]
    )

  def build(self, ctx: plot_utils.BuildContext):
    return None


def test_multi_dashboards_merge_custom_without_in_overview():
  db1 = CustomDashboardNoInOverview(
      name='custom',
      title='Custom Title',
      plots=[plot_utils.Plot(y_key='metric1', collections=['train'])],
  )
  db2 = CustomDashboardNoInOverview(
      name='custom',
      title='Custom Title',
      plots=[plot_utils.Plot(y_key='metric2', collections=['eval'])],
  )
  merged = dashboard_utils.MultiDashboards.from_iterable([db1, db2])
  assert len(merged.dashboards['custom'].plots) == 2
  res = merged.add_overview_dashboard()
  assert 'overview' in res.dashboards
  assert res.dashboards['overview'].plots == [
      plot_utils.Plot(
          y_key='metric1', collections=['train'], remove_prefix=False
      ),
      plot_utils.Plot(
          y_key='metric2', collections=['eval'], remove_prefix=False
      ),
  ]


def test_plot_merge_empty_facet_to_collections():
  # Test merging a plot with empty facets and a plot with plain collections.
  plot_empty_facets = plot_utils.Plot(
      y_key='accuracy',
      collections=[],
      facet_to_collections={' train': [], 'eval': []},
  )
  plot_prefixed_collections = plot_utils.Plot(
      y_key='accuracy',
      collections=['ssv2_classification.eval'],
  )
  merged_plot = plot_utils.Plot.merge(
      [plot_empty_facets, plot_prefixed_collections]
  )
  assert merged_plot.collections == ['ssv2_classification.eval']
  assert not merged_plot.facet_to_collections


def test_plot_merge_single_element_empty_facets():
  plot_empty_facets = plot_utils.Plot(
      y_key='accuracy',
      collections=[],
      facet_to_collections={' train': [], 'eval': []},
  )
  normalized = plot_utils.Plot.merge([plot_empty_facets])
  assert not normalized.facet_to_collections


def test_plot_merge_mixed_faceted_and_unfaceted():
  faceted_plot = plot_utils.Plot(
      y_key='loss',
      collections=['train'],
      facet_to_collections={' train': ['train']},
  )
  plain_plot = plot_utils.Plot(
      y_key='loss',
      collections=['eval'],
  )
  merged = plot_utils.Plot.merge([faceted_plot, plain_plot])
  assert set(merged.collections) == {'train', 'eval'}
  assert merged.facet_to_collections == {' train': ['train'], 'eval': ['eval']}


def test_plot_merge_conflicting_facets_and_ykeys():
  faceted_plot = plot_utils.Plot(
      y_key='loss',
      collections=['train'],
      facet_to_collections={' train': ['train']},
  )
  multi_ykey_plot = plot_utils.Plot(
      y_key='loss',
      collections=['eval'],
      collection_to_ykeys={'eval': ['loss/a', 'loss/b']},
  )
  with pytest.raises(ValueError, match='Cannot merge plot with'):
    plot_utils.Plot.merge([faceted_plot, multi_ykey_plot])
