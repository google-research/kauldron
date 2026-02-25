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

"""Lark-based parser for py:: flag expressions."""

from __future__ import annotations

import ast
import functools
from typing import Any
import warnings

from etils import epath
from kauldron.konfig import configdict_base
from kauldron.konfig import configdict_proxy

# Silence deprecation warnings about sre_parse and sre_constants
with warnings.catch_warnings(action='ignore', category=DeprecationWarning):
  import lark  # pylint: disable=g-import-not-at-top


def parse(expr: str) -> Any:
  """Parses a py:: expression string into ConfigDict objects.

  Supports:
    - Qualnames with explicit colon: `module.path:MyEnum.VALUE` →
        ConfigDict({'__const__': 'module.path:MyEnum.VALUE'})
    - Call expressions: `module.ClassName(x=1)` →
        ConfigDict({'__qualname__': 'module:ClassName', 'x': 1})
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
    tree = _parser().parse(expr)
    return _transformer.transform(tree)
  except lark.LarkError as e:
    raise ValueError(f'Invalid py:: expression: {expr!r}') from e


@functools.cache
def _parser() -> lark.Lark:
  grammar_path = epath.resource_path('kauldron.konfig') / 'py_flag_grammar.lark'
  return lark.Lark(
      grammar=grammar_path.read_text(),
      parser='lalr',
  )


class _PyFlagTransformer(lark.Transformer):
  """Transforms a Lark parse tree into ConfigDict objects and literals."""

  @staticmethod
  def start(args: list[Any]) -> Any:
    return args[0]

  @staticmethod
  def call(args: list[Any]) -> configdict_base.ConfigDict:
    qualname = args[0]
    fields = args[1] if len(args) > 1 and args[1] is not None else {}
    return configdict_base.ConfigDict({
        configdict_proxy.QUALNAME_KEY: qualname,
        **fields,
    })

  @staticmethod
  def arguments(args: list[Any]) -> dict[str, Any]:
    fields = {}
    pos_idx = 0
    for key, value in args:
      if key is None:
        fields[str(pos_idx)] = value
        pos_idx += 1
      else:
        fields[key] = value
    return fields

  @staticmethod
  def kwarg(args: list[Any]) -> tuple[str, Any]:
    return (str(args[0]), args[1])

  @staticmethod
  def posarg(args: list[Any]) -> tuple[None, Any]:
    return (None, args[0])

  @staticmethod
  def const_ref(args: list[Any]) -> configdict_base.ConfigDict:
    return configdict_base.ConfigDict({configdict_proxy.CONST_KEY: args[0]})

  @staticmethod
  def explicit_qualname(args: list[Any]) -> str:
    module_path = args[0]
    attr_path = args[1]
    return configdict_base.expand_qualname(f'{module_path}:{attr_path}')

  @staticmethod
  def implicit_qualname(args: list[Any]) -> str:
    dotted = args[0]
    parts = dotted.split('.')
    if len(parts) < 2:
      raise ValueError(
          f'Invalid py:: qualname: {dotted!r}. Must be at least `module.Name`.'
      )
    qualname = '.'.join(parts[:-1]) + ':' + parts[-1]
    return configdict_base.expand_qualname(qualname)

  @staticmethod
  def dotted_name(args: list[Any]) -> str:
    return '.'.join(str(a) for a in args)

  @staticmethod
  def list_expr(args: list[Any]) -> list[Any]:
    return [a for a in args if a is not None]

  @staticmethod
  def tuple_expr(args: list[Any]) -> tuple[Any, ...]:
    return tuple(args)

  @staticmethod
  def dict_expr(args: list[Any]) -> dict[Any, Any]:
    return dict(a for a in args if a is not None)

  @staticmethod
  def key_value(args: list[Any]) -> tuple[Any, Any]:
    return (args[0], args[1])

  @staticmethod
  def int_literal(args: list[Any]) -> int:
    return int(args[0], 0)

  @staticmethod
  def float_literal(args: list[Any]) -> float:
    return float(args[0])

  @staticmethod
  def imag_literal(args: list[Any]) -> complex:
    return complex(args[0])

  @staticmethod
  def string_literal(args: list[Any]) -> str:
    return ast.literal_eval(str(args[0]))

  @staticmethod
  def true_literal(args: list[Any]) -> bool:
    del args  # Unused
    return True

  @staticmethod
  def false_literal(args: list[Any]) -> bool:
    del args  # Unused
    return False

  @staticmethod
  def none_literal(args: list[Any]) -> None:
    del args  # Unused
    return None

  @staticmethod
  def neg_int(args: list[Any]) -> int:
    return -int(args[0], 0)

  @staticmethod
  def neg_imag(args: list[Any]) -> complex:
    return -complex(args[0])

  @staticmethod
  def neg_float(args: list[Any]) -> float:
    return -float(args[0])


_transformer = _PyFlagTransformer()
