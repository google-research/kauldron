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

"""Plot base class."""

from __future__ import annotations

import dataclasses

from kauldron.utils.kdash import build_utils
from kauldron.utils.kdash import xm_utils
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member


@dataclasses.dataclass(frozen=True, kw_only=True)
class BuildContext:
  """Context for building the dashboard.

  Attributes:
    collection_path_prefix: The `/datatable/users/ldap`, `/s2/xid/1234`,... to
      use. If `None`, is auto-computed.
    sweep_argnames: Sweep arguments from the experiment. Plots uses heuristics
      for example to average plots over seeds. If `None`, is auto-computed.
  """

  collection_path_prefix: str = None  # pytype: disable=annotation-type-mismatch
  sweep_argnames: list[str] = None  # pytype: disable=annotation-type-mismatch

  def __post_init__(self):
    if self.collection_path_prefix is None:
      object.__setattr__(
          self,
          'collection_path_prefix',
          build_utils.make_collection_path_prefix(),
      )

    if self.sweep_argnames is None:  # Auto-detect sweep argnames
      if status.on_xmanager:
        sweep_argnames = xm_utils.get_sweep_argnames(status.xp)
      else:
        sweep_argnames = []
      object.__setattr__(self, 'sweep_argnames', sweep_argnames)

  @property
  def id(self) -> str:
    """ID of the dashboard, can be used in the dashboard title."""
    return (
        self.collection_path_prefix.removeprefix('/datatable/')
        .removeprefix('xid/')
        .removeprefix('users/')
        .removesuffix('.')
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Plot:
  """Single plot inside a dashboard."""

  y_key: str
  x_key: str = 'step'
  collections: list[str]

  remove_prefix: bool = True

  def build(self, ctx: BuildContext) -> fb.Plot:
    """Build the flatboard plot."""
    title = self.y_key.partition('/')[-1] if self.remove_prefix else self.y_key

    # Some heuristics for visualizing the plots.
    # TODO(epot): Add option to override grouping, facet,...

    sweep_argnames = {arg: True for arg in ctx.sweep_argnames}

    # If sweeping over seeds, plot mean+confidence otherwise individual
    has_sweep = sweep_argnames.pop('seed', False)
    if has_sweep:
      transform = fb.DataTransform.MEAN_CONF
    else:
      transform = fb.DataTransform.INDIVIDUAL

    # Separate group for each non-seed sweep-key (if present)
    groups = fb.Groups(columns=[f'HYP_{y_key}' for y_key in sweep_argnames])

    # If both sweeping over values and multiple collections (train / eval)
    # then split collections into facets to prevent crowding the flatboard
    # otherwise combine collections into a single plot
    if sweep_argnames and len(self.collections) > 1:
      facets = fb.Facets(
          columns=['collection'],
          couple_scales=True,
          num_cols=len(self.collections),
      )
      # If doing a facet over collections, then:
      # - we need to set the collections argument for each datagroup. In that
      #   case we also rename "train" to " train" as a hack to ensure that it is
      #   displayed first.
      # - we do not add any name to the data group to keep the labels short
      #   (e.g. "lr=0.1" instead of "train, lr=0.1")
      data_groups = [
          fb.DataGroup(  # pylint: disable=g-complex-comprehension
              name='',
              queries=[
                  fb.DataQuery(
                      query=f'{ctx.collection_path_prefix}{c}',
                      set={'collection': ' train' if c == 'train' else c},
                  )
                  for c in self.collections
              ],
          )
      ]
    else:
      # We are not faceting over collections, so we add one data group per
      # collection with the appropriate name. That way the labels will be
      # "train", "eval" etc.
      facets = fb.Facets()
      data_groups = [
          fb.DataGroup(
              name=c,
              queries=[fb.DataQuery(query=f'{ctx.collection_path_prefix}{c}')],
          )
          for c in self.collections
      ]

    return fb.Plot(
        title=title,
        x_key=self.x_key,
        y_key=self.y_key,
        data_groups=data_groups,
        transform=transform,
        groups=groups,
        facets=facets,
    )
