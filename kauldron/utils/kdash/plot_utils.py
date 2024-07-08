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

import collections
import dataclasses
import itertools

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
  """Single plot inside a dashboard.

  Attributes:
    y_key: Metric to plot.
    x_key: `step` by default.
    collections: List of collections to plot.
    facet_to_collections: Group the collections by facet. Each facet is
      displayed as a separate plot. If set, `collections=` is optional.
    collection_to_ykeys: Allow to display multiple `y_keys` from the same
      collection together. If set, `collections=` is optional.
    remove_prefix: Remove the `losses/`, `metrics/`,... prefixes.
  """

  y_key: str
  x_key: str = 'step'
  collections: list[str] = dataclasses.field(default_factory=list)
  facet_to_collections: dict[str, list[str]] = dataclasses.field(
      default_factory=dict
  )
  collection_to_ykeys: dict[str, list[str]] = dataclasses.field(
      default_factory=dict
  )

  remove_prefix: bool = True

  def __post_init__(self):
    # Normalize the values.
    if self.facet_to_collections:
      assert not self.collection_to_ykeys
      collections_ = itertools.chain.from_iterable(
          self.facet_to_collections.values()
      )
      if self.collections and set(collections_) != set(self.collections):
        raise ValueError(
            'The `collections` and `facet_to_collections` values are not'
            f' consistent. {self.collections} vs {self.facet_to_collections}'
        )
      if not self.collections:
        object.__setattr__(self, 'collections', list(collections_))
    # Normalize the values.
    if self.collection_to_ykeys:
      assert not self.facet_to_collections
      collections_ = list(self.collection_to_ykeys)
      if self.collections and set(collections_) != set(self.collections):
        raise ValueError(
            'The `collections` and `collection_to_ykeys` values are not'
            f' consistent. {self.collections} vs'
            f' {collections_}'
        )
      if not self.collections:
        object.__setattr__(self, 'collections', list(collections_))

  @classmethod
  def merge(cls, plots: list[Plot]) -> Plot:
    """Merges multiple plots with the same y_key."""
    assert len(plots)  # pylint: disable=g-explicit-length-test
    if len(plots) == 1:
      return plots[0]
    # Cannot merge plots with different x_key or y_key.
    # TODO(epot): Signature should be a plot property
    signature = set((p.x_key, p.y_key) for p in plots)
    if len(signature) != 1:
      raise ValueError(
          f'Cannot merge plots with different x_key or y_key: {signature}'
      )

    # TODO(epot): Remove duplicates while keeping order.
    merged_collections = list(
        itertools.chain.from_iterable(p.collections for p in plots)
    )
    merged_facet_to_collections = collections.defaultdict(list)
    merged_collection_to_ykeys = collections.defaultdict(list)
    for p in plots:
      for facet, collections_ in p.facet_to_collections.items():
        merged_facet_to_collections[facet].extend(collections_)
      for collection, ykeys in p.collection_to_ykeys.items():
        merged_collection_to_ykeys[collection].extend(ykeys)

    # TODO(epot): Generic validation which check all fields except a list match
    return dataclasses.replace(
        plots[0],
        collections=merged_collections,
        facet_to_collections=dict(merged_facet_to_collections),
        collection_to_ykeys=dict(merged_collection_to_ykeys),
    )

  def build(self, ctx: BuildContext) -> fb.Plot:
    """Build the flatboard plot."""
    title = self._normalize_key(self.y_key)

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

    if self.facet_to_collections:
      facet_to_collections = self.facet_to_collections
    elif sweep_argnames and len(self.collections) > 1:
      # If both sweeping over values and multiple collections (train / eval)
      # then split collections into facets to prevent crowding the flatboard
      # otherwise combine collections into a single plot
      # Rename "train" to " train" as a hack to ensure that it is displayed
      # first.
      facet_to_collections = {
          (' train' if c == 'train' else c): [c] for c in self.collections
      }
    else:
      facet_to_collections = {}

    if facet_to_collections:
      facets = fb.Facets(
          columns=['collection'],
          couple_scales=True,
          num_cols=len(facet_to_collections),
      )
      queries = []
      for facet_name, collections_ in facet_to_collections.items():
        for collection in collections_:
          query = fb.DataQuery(
              query=f'{ctx.collection_path_prefix}{collection}',
              set={'collection': facet_name},
          )
          queries.append(query)
      data_groups = [
          fb.DataGroup(  # pylint: disable=g-complex-comprehension
              # We do not add any name to the data group to keep the labels
              # short (e.g. "lr=0.1" instead of "train, lr=0.1")
              name='',
              queries=queries,
          )
      ]
    elif self.collection_to_ykeys:
      facets = fb.Facets()
      data_groups = []
      for c, ykeys in self.collection_to_ykeys.items():
        for ykey in ykeys:
          data_group = fb.DataGroup(
              name=f'{c}/{self._normalize_key(ykey)}',
              queries=[
                  fb.DataQuery(
                      query=f'{ctx.collection_path_prefix}{c}',
                      ykey=ykey,
                  )
              ],
          )
          data_groups.append(data_group)

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

  def _normalize_key(self, key: str) -> str:
    """Normalize the key to remove the prefix."""
    if self.remove_prefix:
      return key.partition('/')[-1]
    else:
      return key
