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

from typing import Any

from kauldron.konfig import py_flag_parser

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
    - Explicit colon qualnames: `module.path:MyEnum.VALUE` →
        ConfigDict({'__const__': 'module.path:MyEnum.VALUE'})
    - Lists: `[module.X(), 4]` → [ConfigDict({...}), 4]
    - Tuples: `(module.X(), 4)` → (ConfigDict({...}), 4)
    - Dicts: `{'key': module.X()}` → {'key': ConfigDict({...})}
    - Literals: `1`, `'hello'`, `True`, `None`

  Args:
    expr: A Python expression string.

  Returns:
    A parsed value: ConfigDict, list, tuple, dict, or a literal.
  """
  return py_flag_parser.parse(expr)
