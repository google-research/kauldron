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

"""Graphviz utils."""

from __future__ import annotations

import collections
from collections.abc import Iterable
import dataclasses

from etils import epy
from kauldron import kontext
from kauldron.train import trainer_lib

with epy.lazy_imports():
  import graphviz  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


def get_connection_graph(trainer: trainer_lib.Trainer) -> graphviz.Digraph:
  """Build the graphviz."""
  nodes = _extract_graph(trainer)
  return _make_graph(nodes)


@dataclasses.dataclass(kw_only=True)
class _Node:
  """Graph node."""

  name: str = None  # pytype: disable=annotation-type-mismatch
  group: str = None  # pytype: disable=annotation-type-mismatch
  inputs: dict[str, str] = dataclasses.field(default_factory=dict)
  outputs: set[str] = dataclasses.field(default_factory=set)

  @property
  def html_table(self) -> str:
    rows = self.inputs.keys() | self.outputs
    rows = sorted(rows - {'step'})
    return _make_node(self.name, rows)

  @property
  def edges(self) -> Iterable[tuple[str, str]]:
    for k, v in self.inputs.items():
      if v == 'step':
        continue
      yield f'{_path_to_graphviz_path(v)}', f'{self.name}:{k}'


def _extract_graph(trainer: trainer_lib.Trainer) -> list[_Node]:
  """Extract the nodes from the trainer."""
  # TODO(epot): Would be nice to add links to the source code.
  # TODO(epot): Filter optional values (mask, step)

  nodes = []
  for group_name, group_objs in {
      'train': {'model': trainer.model},
      'losses': dict(trainer.train_losses),
      'metrics': dict(trainer.train_metrics),
      'summaries': dict(trainer.train_summaries),
  }.items():
    for obj_name, keyed_obj in group_objs.items():
      obj_inputs = kontext.get_keypaths(keyed_obj)
      # Should likely flatten here to support nested kontext
      node = _Node(
          name=obj_name,
          group=group_name,
          inputs={k: v for k, v in obj_inputs.items() if v is not None},
      )
      nodes.append(node)

  additional_nodes = collections.defaultdict(_Node)
  for node in nodes:
    for k, v in node.inputs.items():
      if '.' in v:
        node_name, rest = v.split('.', 1)
      else:
        node_name, rest = k, None
      additional_nodes[node_name].name = node_name
      additional_nodes[node_name].group = 'train'
      if rest is not None:
        additional_nodes[node_name].outputs.add(rest)
  additional_nodes.pop('step', None)

  nodes.extend(additional_nodes.values())
  return nodes


def _make_graph(nodes: list[_Node]) -> graphviz.Digraph:
  """Build the graphviz object."""
  group_to_node = epy.groupby(nodes, key=lambda x: x.group)

  dot = graphviz.Digraph()

  # TODO(epot): How to better style the graph ?
  dot.attr(rankdir='LR')
  dot.attr('node', shape='none')

  for group, nodes in group_to_node.items():
    with dot.subgraph(name=f'cluster_{group}') as subgraph:
      subgraph.attr(label=group)
      subgraph.attr(rankdir='TB')

      for node in nodes:
        # Make a node for the current object
        subgraph.node(node.name, label=node.html_table)

        # Make a node for the model
        for from_, to in node.edges:
          subgraph.edge(from_, to)

      if group == 'train':
        subgraph.edge('model', 'preds')
  return dot


def _make_node(title: str, rows: Iterable[str]) -> str:
  lines = [H.tr(H.td(H.b(title)))]
  lines.extend(H.tr(H.td(k, port=_normalize_label(k))) for k in rows)
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
  return f'{prefix}:{_normalize_label(suffix)}'


def _normalize_label(label: str) -> str:
  for invalid_char in ',: []':
    label = label.replace(invalid_char, '_')
  return label


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
