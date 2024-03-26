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

"""Tools for string paths such as "cfg.net.layers[0].act_fun"."""

from __future__ import annotations

import ast
import enum
import functools
from types import EllipsisType  # pylint: disable=g-importing-member
from typing import Any, Optional, Union

from etils import epath
from etils import epy
import lark


class Wildcard(enum.StrEnum):
  """Wildcard types."""

  STAR = "*"
  DOUBLE_STAR = "**"


Part = (
    int
    | float
    | complex
    | slice
    | str
    | bool
    | None
    | EllipsisType
    | tuple
    | Wildcard
)


def parse_parts(str_path: str) -> list[Part]:
  """Returns a list of PathParts from a string."""
  parser = _path_parser()
  transformer = _PathTransformer()
  try:
    tree = parser.parse(str_path)
    parts = transformer.transform(tree)
  except Exception as e:  # pylint: disable=broad-exception-caught
    epy.reraise(e, f"Could not parse path: {str_path!r}: ")
  else:
    return parts


@functools.cache
def _path_parser() -> lark.Lark:
  grammar_path = epath.resource_path("kauldron.kontext") / "path_grammar.lark"
  return lark.Lark(
      start="path",
      regex=True,
      grammar=grammar_path.read_text(),
  )


class _PathTransformer(lark.Transformer):
  """Transforms a Lark parse-tree into a Path object."""

  @staticmethod
  def path(args: list[Any]) -> list[Part]:
    return args

  @staticmethod
  def IDENTIFIER(args: list[lark.Token]) -> str:
    assert isinstance(args, str)
    return args

  @staticmethod
  def slice_key(args: list[int | str | None]) -> slice:
    sargs: list[Optional[int]] = [None, None, None]
    i = 0
    for a in args:
      if a == ":":
        i += 1
      else:
        sargs[i] = int(a) if a is not None else None
    return slice(*sargs)

  @staticmethod
  def ellipsis(_):
    return ...

  @staticmethod
  def tensor_slice_key(args: list[Any]) -> Any:
    return tuple(args)

  @staticmethod
  def number(args: list[str]) -> Union[int, float, complex]:
    return ast.literal_eval(args[0])

  @staticmethod
  def integer(args: list[str]) -> int:
    return ast.literal_eval(args[0])

  @staticmethod
  def none(_) -> None:
    return None

  @staticmethod
  def boolean(args: list[str]) -> bool:
    return {"True": True, "False": False}[args[0]]

  @staticmethod
  def string(args: list[str]) -> str:
    return args[0][1:-1]

  @staticmethod
  def tuple_key(args: list[Any]) -> tuple[Any, ...]:
    return tuple(args)

  @staticmethod
  def WS(_):
    # discard all whitespace tokens
    raise lark.Discard()

  @staticmethod
  def STAR(_):
    return Wildcard.STAR

  @staticmethod
  def DOUBLE_STAR(_):
    return Wildcard.DOUBLE_STAR
