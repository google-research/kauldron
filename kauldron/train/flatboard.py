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

"""Flatboard."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Optional, Sequence

from kauldron.train.status_utils import status  # pylint: disable=g-importing-member

xmanager_api = _mock.Mock()
fb = _mock.Mock()


class DashboardFactory(abc.ABC):

  def __call__(self) -> fb.Dashboard | None:
    raise NotImplementedError()


@dataclasses.dataclass
class DefaultDashboard(DashboardFactory):
  """Create a basic flatboard dashboard for current experiment.

  Will generate different dashboards depending on whether the experiment is a
  hyperparameter sweep. In particular:
    - for a single run the plots for each metric combine single lines for each
      collection (e.g. train and eval).
    - if the experiment sweeps over the seed hyperparameter, then a
      mean + confidence transformation is applied to each metric.
    - if the experiment sweeps over non-seed hyperparameters, then a group is
      added for each parameter, and the collections are separated into facets.
      E.g. train and eval are two separate plots with one line per hyperparam.

  Attributes:
    title: The title template for the dashboard (as shown in the flatboard
      browser tab). This can contain {xid} and {experiment_name} placeholders.
    y_keys: The y_keys (metrics) for which to create individual plots.
    collections: The datatable collections to include. E.g. train and eval.
    remove_prefix: Whether to drop the part before the first '/' of the y_key
      for the title of the individual plot. E.g. 'total' instead of
      'losses/total'.
  """

  title: str = "{xid}: {experiment_name}"  # name shown in browser tab
  y_keys: Sequence[str] = ()
  collections: Sequence[str] = ("train",)
  remove_prefix: bool = True

  def __call__(self) -> Optional[fb.Dashboard]:
    if not status.on_xmanager or not status.is_lead_host:
      return  # only add the flatboard once (on the lead host)

    xp = xmanager_api.XManagerApi().get_current_experiment()

    # if sweeping over seeds, plot mean+confidence otherwise individual
    sweep_info = get_sweep_info(xp)
    seed_values = sweep_info.pop("seed", ())
    if len(seed_values) > 1:
      transform = fb.DataTransform.MEAN_CONF
    else:
      transform = fb.DataTransform.INDIVIDUAL

    # separate group for each non-seed sweep-key (if present)
    groups = [y_key for y_key in sweep_info]

    # if both sweeping over values and multiple collections (train / eval)
    # then split collections into facets to prevent crowding the flatboard
    # otherwise combine collections into a single plot
    facets = ["collection"] if groups and len(self.collections) > 1 else []

    metrics = {self._maybe_remove_prefix(y): ("step", y) for y in self.y_keys}

    return create_dashboard(
        xp=xp,
        title=self.title,
        metrics=metrics,
        collections=self.collections,
        groups=groups,
        facets=facets,
        transform=transform,
    )

  def _maybe_remove_prefix(self, y: str) -> str:
    return y.partition("/")[-1] if self.remove_prefix else y


def get_sweep_info(xp: xmanager_api.Experiment) -> dict[str, set[Any]]:
  """Return a mapping sweep-keys to a set of their values for given XID.

  Sweep-keys are all keys that are set in any of the work-units, and which do
  not have a constant value.
  Args:
    xp: The xmanager experiment to get the sweep info for.

  Returns:
    A dictionary with each sweep-key and the set of corresponding values that
    key takes among any work unit.
  """
  # raw sweep is just a list of config overrides for all work-units
  wu_params = [wu.parameters for wu in xp.get_work_units()]
  num_work_units = len(wu_params)
  # merge them by keys into a single tree with a list of possible values
  all_override_keys = {k for wup in wu_params for k in wup}  # pylint: disable=g-complex-comprehension
  zipped_sweep_values = {
      k: [params[k] for params in wu_params if k in params]
      for k in all_override_keys
  }
  # filter out simple overrides (non-sweep keys), which are characterized by:
  # 1. they occur for every wu and 2. they always have the same value
  filtered_sweep_set = {
      k: set(v)
      for k, v in zipped_sweep_values.items()
      if len(v) < num_work_units or len(set(v)) > 1
  }
  return filtered_sweep_set


# TODO(klausg) make individual plots configurable from the config
# TODO(klausg) maybe make individual plots configurable from the metrics?
def create_dashboard(
    xp: xmanager_api.Experiment,
    metrics: dict[str, tuple[str, str]],
    title: str = "{xid}: {experiment_name}",
    collections: Sequence[str] = ("train",),
    groups: Sequence[str] = (),
    facets: Sequence[str] = (),
    transform=fb.DataTransform.INDIVIDUAL,
) -> fb.Dashboard:
  """Create a flatboard dashboard.

  Args:
    xp: The xmanager experiment.
    metrics: A dictionary mapping the name of all plots to (x_key, y_key).
    title: The title template for the dashboard (as shown in the flatboard
      browser tab). This can contain {xid} and {experiment_name} placeholders.
    collections: A list of datatable collections to use for the plots.
    groups: A list of data columns to group. See group argument of fb.Plot.
    facets: A list of data columns to divide into facets. See facets argument of
      fb.Plot.
    transform: See transform argument of fb.Plot.

  Returns:
    A fb.Dashboard instance with one plot for each metric.
  """
  title = title.format(xid=xp.id, experiment_name=xp.name)
  data_groups = [
      fb.DataGroup(  # pylint: disable=g-complex-comprehension
          name=coll,
          queries=[
              fb.DataQuery(
                  query=f"/datatable/xid/{xp.id}/{coll}",
                  set={"collection": coll} if "collection" in facets else {},
              )
          ],
      )
      for coll in collections
  ]

  return fb.Dashboard(
      title=title,
      plots=[
          fb.Plot(  # pylint: disable=g-complex-comprehension
              title=name,
              x_key=x,
              y_key=y,
              transform=transform,
              data_groups=data_groups,
              facets=fb.Facets(
                  columns=list(facets),
                  couple_scales=True,
                  num_cols=len(collections),
              ),
              groups=fb.Groups(columns=[f"HYP_{y_key}" for y_key in groups]),
          )
          for name, (x, y) in metrics.items()
      ],
  )


def _collect_datatables(dashboard: fb.Dashboard) -> set[str]:
  data_sources = set()
  for plot in dashboard.plots:
    for data_group in plot.data_groups:
      for query in data_group.queries:
        if query.query.startswith("/datatable"):
          data_sources.add(query.query)
  return {q.partition(":")[0] for q in data_sources}


def add_flatboard_artifacts(dashboard_factories: dict[str, DashboardFactory]):
  """Add flatboard artifacts and corresponding datatable artifacts to XM."""
  xp = xmanager_api.XManagerApi().get_current_experiment()
  data_sources = set()
  for name, factory in dashboard_factories.items():
    dashboard = factory()
    if dashboard:
      data_sources |= _collect_datatables(dashboard)
      xp.create_artifact(
          xmanager_api.ArtifactType.ARTIFACT_TYPE_FLATBOARD_URL,
          dashboard.save_url(
              reader_permissions=fb.FlatboardDashboardPermissions.EVERYONE
          ).split("/revisions/")[0],
          name,
      )
  # add artifacts for the datatables
  for data_source in data_sources:
    xp.create_artifact(
        artifact_type=xmanager_api.ArtifactType.ARTIFACT_TYPE_STORAGE2_BIGTABLE,
        artifact=data_source,
        description=data_source.rpartition("/")[-1],
    )
