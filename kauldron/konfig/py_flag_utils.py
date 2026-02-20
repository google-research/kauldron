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

"""Utilities for parsing `py::` flag values into ConfigDict objects."""

from __future__ import annotations

import ast
from typing import Any

from kauldron.konfig import configdict_base
from kauldron.konfig import configdict_proxy

_PY_PREFIX = 'py::'


def maybe_parse_py_flag_value(value: Any) -> Any:
  """Parses value if it is a `py::` prefixed string, otherwise returns as-is."""
  if not isinstance(value, str) or not value.startswith(_PY_PREFIX):
    return value
  return parse_py_flag_value(value.removeprefix(_PY_PREFIX))


def parse_py_flag_value(expr: str) -> Any:
  """Parses a Python expression string into ConfigDict objects.

  Supports:
    - Call expressions: `module.ClassName(x=1)` →
        ConfigDict({'__qualname__': 'module:ClassName', 'x': 1})
    - Attribute references: `module.ClassName` →
        ConfigDict({'__const__': 'module:ClassName'})
    - Lists: `[module.X(), 4]` → [ConfigDict({...}), 4]
    - Tuples: `(module.X(), 4)` → (ConfigDict({...}), 4)
    - Dicts: `{'key': module.X()}` → {'key': ConfigDict({...})}
    - Literals: `1`, `'hello'`, `True`, `None`

  Args:
    expr: A Python expression string.

  Returns:
    A parsed value: ConfigDict, list, tuple, dict, or a literal.
  """
  try:
    tree = ast.parse(expr, mode='eval')
  except SyntaxError as e:
    raise ValueError(f'Invalid py:: expression: {expr!r}') from e
  return _convert_node(tree.body)


def _convert_node(node: ast.expr) -> Any:
  """Converts an AST node to a ConfigDict or literal value."""
  match node:
    case ast.Call():
      return _convert_call(node)
    case ast.Attribute() | ast.Name():
      return _convert_const_ref(node)
    case ast.List():
      return [_convert_node(elt) for elt in node.elts]
    case ast.Tuple():
      return tuple(_convert_node(elt) for elt in node.elts)
    case ast.Dict():
      return {
          _convert_node(k): _convert_node(v)
          for k, v in zip(node.keys, node.values)
      }
    case ast.Constant():
      return node.value
    case ast.UnaryOp(
        op=ast.USub(), operand=ast.Constant(value=v)
    ) if isinstance(v, int | float):
      return -v
    case _:
      raise ValueError(f'Unsupported py:: expression node: {ast.dump(node)}')


def _convert_call(node: ast.Call) -> configdict_base.ConfigDict:
  """Converts an `ast.Call` node to a ConfigDict with `__qualname__`."""
  qualname = _node_to_qualname(node.func)

  fields = {configdict_proxy.QUALNAME_KEY: qualname}
  for i, arg in enumerate(node.args):
    fields[str(i)] = _convert_node(arg)
  for kw in node.keywords:
    fields[kw.arg] = _convert_node(kw.value)
  return configdict_base.ConfigDict(fields)


def _convert_const_ref(node: ast.expr) -> configdict_base.ConfigDict:
  """Converts an attribute chain to a ConfigDict with `__const__`."""
  qualname = _node_to_qualname(node)
  return configdict_base.ConfigDict({configdict_proxy.CONST_KEY: qualname})


def _node_to_qualname(node: ast.expr) -> str:
  """Converts an AST name/attribute chain to a qualname string.

  `module.path.ClassName` → `module.path:ClassName`

  The convention is: everything before the last `.` is the module path,
  and the last dotted segment is the attribute name, separated by `:`.

  Args:
    node: An AST Name or Attribute node.

  Returns:
    A qualname string like `module.path:ClassName`.

  Raises:
    ValueError: If the node is not a valid name/attribute chain.
  """
  parts = _collect_name_parts(node)
  if len(parts) < 2:
    raise ValueError(
        f'Invalid py:: qualname: {".".join(parts)!r}. Must be at least'
        ' `module.Name`.'
    )
  return '.'.join(parts[:-1]) + ':' + parts[-1]


def _collect_name_parts(node: ast.expr) -> list[str]:
  """Collects dotted name parts from an AST node chain."""
  match node:
    case ast.Name(id=name):
      return [name]
    case ast.Attribute(value=value, attr=attr):
      return _collect_name_parts(value) + [attr]
    case _:
      raise ValueError(f'Unsupported py:: qualname node: {ast.dump(node)}')
