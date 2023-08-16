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

"""Some plotting helpers."""
from __future__ import annotations

import altair as alt
import jax
from kauldron.typing import PyTree, Schedule  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import paths
import ml_collections
import numpy as np
import pandas as pd


def plot_schedules(schedules: PyTree[Schedule], num_steps: int) -> alt.Chart:
  """Overview plot for (nested) dict of schedules."""
  if isinstance(schedules, ml_collections.ConfigDict):
    schedules = schedules.to_dict()
  # flatten schedules
  flat_schedules = paths.tree_flatten_with_path(schedules)
  # evaluate each for 1000 linearly spaced step-values
  x = np.round(np.linspace(0, num_steps, num=1000)).astype(int)
  sched_values = {
      name: jax.vmap(sched)(x) for name, sched in flat_schedules.items()
  }
  # build a dataframe from this
  rows = []
  for name, values in sched_values.items():
    for step, v in zip(x, values):
      rows.append({"Step": int(step), "Schedule": str(name), "Value": float(v)})
  schedules_df = pd.DataFrame(rows)

  # construct an Altair plot
  highlight = alt.selection(
      type="single", on="mouseover", fields=["Schedule"], nearest=True
  )
  base = alt.Chart(schedules_df).encode(
      x="Step:Q",
      y="Value:Q",
      color="Schedule:N",
      tooltip=["Schedule", "Step", "Value"],
  )
  points = (
      base.mark_circle()
      .encode(opacity=alt.value(0))
      .properties(selection=highlight, width=600)
  )
  lines = base.mark_line().encode(
      size=alt.condition(~highlight, alt.value(1), alt.value(3))
  )

  return points + lines
