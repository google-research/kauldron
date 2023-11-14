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

"""Graphviz utils."""

from __future__ import annotations

from collections.abc import Iterable

from etils import epy
from kauldron import kontext
from kauldron.train import config_lib

with epy.lazy_imports():
  import graphviz  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


def get_connection_graph(cfg: config_lib.Config) -> graphviz.Digraph:
  """Build the graphviz."""
  dot = graphviz.Digraph()

  # TODO(epot): Should refactor to avoid circular dependency
  from kauldron.utils import inspect  # pylint: disable=g-import-not-at-top

  ctx = inspect.eval_context_shape(cfg)

  # TODO(epot): How to better style the graph ?
  dot.attr(rankdir='LR')
  dot.attr('node', shape='none')

  with dot.subgraph(name='cluster_train') as subgraph:
    # Make a node for the batch
    batch = list(kontext.flatten_with_path(ctx.batch).keys())
    subgraph.node('batch', label=_make_node('batch', batch))

    # Make a node for the model
    model_inputs = kontext.get_keypaths(cfg.model)
    subgraph.node('model', label=_make_node('model', model_inputs.keys()))
    for k, v in model_inputs.items():
      subgraph.edge(f'{_path_to_graphviz_path(v)}', f'model:{k}')

    # Make a node for the model preds
    preds = list(kontext.flatten_with_path(ctx.preds).keys())
    subgraph.node('preds', label=_make_node('preds', preds))
    subgraph.edge('model', 'preds')

  # TODO(epot): Would be nice to add links to the source code.
  # TODO(epot): Filter optional values (mask)

  # Make a node for each losses, metrics, summaries,...
  for group_name, group_objs in {
      'train': {'model': cfg.model},
      'losses': dict(cfg.train_losses),
      'metrics': dict(cfg.train_metrics),
      'summaries': dict(cfg.train_summaries),
  }.items():
    with dot.subgraph(name=f'cluster_{group_name}') as subgraph:
      subgraph.attr(label=group_name)
      subgraph.attr(rankdir='TB')
      for obj_name, keyed_obj in group_objs.items():
        obj_inputs = kontext.get_keypaths(keyed_obj)
        subgraph.node(obj_name, label=_make_node(obj_name, obj_inputs.keys()))
        for k, v in obj_inputs.items():
          if v is None:
            continue
          subgraph.edge(f'{_path_to_graphviz_path(v)}', f'{obj_name}:{k}')

  # Connect everything together

  return dot


def _make_node(title: str, rows: Iterable[str]) -> str:
  lines = [H.tr(H.td(H.b(title)))]
  lines.extend(H.tr(H.td(k, port=k)) for k in rows)
  table = H.table(
      *lines,
      border='0',
      cellborder='1',
      cellspacing='0',
      # style='background-color:#f5f5f5;',  # Looks css isn't supported
  )
  return f'<{table.build()}>'


def _path_to_graphviz_path(k: str) -> str:
  if '.' not in k:
    return k
  prefix, suffix = k.split('.', 1)
  return f'{prefix}:{suffix}'


# ------------ Mini HTML helper lib ------------


class _Str(str):

  def __repr__(self) -> str:  # pylint: disable=invalid-repr-returned
    return self


class _HTMLTag:
  """HTML tag."""

  name: str

  def __init__(self, *content: str | _HTMLTag, **attributes: str) -> None:
    self._content = content
    self._attributes = attributes

  def build(self) -> str:
    if self._attributes:
      attributes = [f'{k}="{v}"' for k, v in self._attributes.items()]
      attributes = ' '.join(attributes)
      attributes = ' ' + attributes
    else:
      attributes = ''
    return epy.Lines.make_block(
        content=_Str(_normalize_content(self._content)),
        braces=(f'<{self.name}{attributes}>', f'</{self.name}>'),
    )


def _normalize_content(content) -> str:
  if isinstance(content, _HTMLTag):
    return content.build()
  elif isinstance(content, (list, tuple)):
    return ''.join(_normalize_content(c) for c in content)
  else:
    return content


class _HTMLTagsBuilder:
  """Dynamically create HTML tags."""

  def __getattr__(self, key: str) -> type[_HTMLTag]:
    class _CurrTag(_HTMLTag):
      name: str = key

    _CurrTag.__name__ = f'{key.capitalize()}Tag'
    return _CurrTag


H = _HTMLTagsBuilder()
