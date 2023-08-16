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

"""Tools for string paths such as "cfg.net.layers[0].act_fun"."""
from __future__ import annotations

import ast
import collections
from typing import Any, Optional, Union, overload

from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import jax.tree_util
from kauldron.typing import PyTree  # pylint: disable=g-importing-member
import lark


JaxKeyEntry = Union[
    jax.tree_util.SequenceKey,
    jax.tree_util.DictKey,
    jax.tree_util.GetAttrKey,
    jax.tree_util.FlattenedIndexKey,
]
Part = int | float | complex | slice | str | bool | None


class Path(collections.abc.Sequence):
  """Represents a string path."""

  __slots__ = ("parts",)

  def __init__(self, *parts: Part):
    if not all(isinstance(part, Part) for part in parts):
      raise ValueError(f"invalid part(s) {parts}")
    self.parts: tuple[Part, ...] = parts

  @overload
  def __getitem__(self, key: int) -> Part:
    ...

  @overload
  def __getitem__(self, key: slice) -> Path:
    ...

  def __getitem__(self, key: int | slice) -> Part | Path:
    if isinstance(key, int):
      return self.parts[key]
    elif isinstance(key, slice):
      return type(self)(*self.parts[key])
    else:
      raise KeyError(f"Invalid key={key!r}: must be int or slice.")

  def __len__(self) -> int:
    return len(self.parts)

  def __hash__(self) -> int:
    hashable_parts = tuple(
        p if not isinstance(p, slice) else (slice, p.start, p.stop, p.step)
        for p in self.parts
    )
    return hash(hashable_parts)

  def __eq__(self, other: Any) -> bool:
    if isinstance(other, Path):
      return self.parts == other.parts
    else:
      return False

  def __repr__(self) -> str:
    r = "".join(_format_part(part) for part in self.parts)
    r = r.removeprefix(".")
    return r

  @classmethod
  def from_str(cls, str_path: str) -> Path:
    tree = _path_parser.parse(str_path)
    new_path = _PathTransformer().transform(tree)
    return new_path

  @classmethod
  def from_jax_path(cls, jax_path: tuple[JaxKeyEntry, ...]) -> Path:
    """Create Path object from a jax path.

    Args:
      jax_path: A jax key path for example from
        `jax.tree_util.tree_flatten_with_path(...)`.

    Returns:
      The corresponding kd.core.Path.
    """
    return cls(*(_jax_key_entry_to_kd_path_element(p) for p in jax_path))

  # TODO(klausg): docstring, annotations and better name
  def get_from(
      self,
      context,
      *,
      default=...,
      err_spec: bool = False,
  ):
    """Extract the object from the path."""
    result = context
    for part in self.parts:
      try:
        result = result[part]
      except (TypeError, IndexError, KeyError):
        if hasattr(result, part):
          result = getattr(result, part)
        else:
          if default is ...:
            # If this fail, allow to have better error message (display the
            # structure)
            if err_spec:
              struct = etree.spec_like(context)
              struct = f"\nContext structure: {epy.pretty_repr(struct)}"
            else:
              struct = ""
            raise KeyError(
                f"Could not find path: {self} in {type(context)}."
                f" {type(result)} has no attributes/key {part}.{struct}"
            ) from None
          return default
    return result


def _jax_key_entry_to_kd_path_element(
    jax_key_entry: JaxKeyEntry,
) -> Part:
  """Convert a JaxKeyEntry into a valid kd.core.Path element."""
  # pytype: disable=match-error
  match jax_key_entry:
    case jax.tree_util.GetAttrKey(name):
      return name
    case jax.tree_util.DictKey(key):
      if not isinstance(key, Part):
        raise TypeError(f"Invalid dict key type {type(key)}")
      return key
    case jax.tree_util.SequenceKey(idx):
      return idx
    case jax.tree_util.FlattenedIndexKey(key):
      return key
  # pytype: enable=match-error
  raise TypeError(f"Unknown key entry type {type(jax_key_entry)}")


def jax_key_entry_to_str(jax_key_entry: JaxKeyEntry) -> str:
  return str(_jax_key_entry_to_kd_path_element(jax_key_entry))


def get_by_path(obj: Any, path: str | tuple[str] | Path, default=None) -> Any:
  """Get (nested) item or attribute by given path.

  Args:
    obj: The object / dict / dataclass to retrieve the value from.
    path: The path to retrieve, either as a string "foo[1]", a tuple ("foo", 1)
      or a Path object.
    default: return value if no value is found.

  Returns:
    The value corresponding to the path if it exists, or `default` otherwise.

  Raises:
    TypeError: if the path is not `str`, `tuple` or `Path`.
  """
  match path:
    case str() if hasattr(obj, path):
      return getattr(obj, path)  # If has attribute, then access directly.
    case str():
      path = Path.from_str(path)  # Otherwise, try parsing key as path.
    case tuple() as parts:
      path = Path(*parts)
    case Path():
      pass
    case None:  # If path is None return the default (useful for optional paths)
      return default
    case _:
      raise TypeError(f"Unknown key/path {path} of type{type(path)}")

  return path.get_from(obj, default=default)


def tree_flatten_with_path(pytree: PyTree[Any]) -> dict[str, Any]:
  flat_tree_items, _ = jax.tree_util.tree_flatten_with_path(pytree)
  return {
      str(Path.from_jax_path(jax_path)): value
      for jax_path, value in flat_tree_items
  }


def _format_part(part: Any) -> str:
  if isinstance(part, str) and part.isidentifier():
    return "." + part
  elif isinstance(part, slice):
    fm = [part.start, ":", part.stop]
    if part.step is not None:
      fm += [":", part.step]
    slice_str = "".join(str(f) for f in fm if f is not None)
    return f"[{slice_str}]"
  else:
    return f"[{part!r}]"


_path_parser = lark.Lark(
    start="path",
    regex=True,
    grammar=r"""
// A path is a series of dot-separated identifiers and [] based item-access.
path: [(identifier | "[" key "]") ("." identifier | "[" key "]")*]
?key: number   // item-access keys can be any hashable python literal
    | slice_key
    | boolean
    | none
    | string
    | tuple_key

tuple_key: "()"
         | "(" key ",)"
         | "(" key ("," key)+ [","] ")"

number: DEC_NUMBER
      | HEX_NUMBER
      | BIN_NUMBER
      | OCT_NUMBER
      | FLOAT_NUMBER
      | COMPLEX_NUMBER

?integer: DEC_NUMBER
       | HEX_NUMBER
       | BIN_NUMBER
       | OCT_NUMBER

!slice_key: [integer] ":" [integer]
          | [integer] ":" [integer] ":" [integer]
string: /".*?(?<!\\)(\\\\)*?"/ | /'.*?(?<!\\)(\\\\)*?'/
!none: "None"
!boolean: "True" | "False"

identifier: IDENTIFIER
IDENTIFIER: ID_START ID_CONTINUE*
ID_START: /[\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nl}_]+/
ID_CONTINUE: ID_START | /[\p{Mn}\p{Mc}\p{Nd}\p{Pc}·]+/

DEC_NUMBER: /-?\d+/
HEX_NUMBER: /-?0x[\da-f]*/i
OCT_NUMBER: /-?0o[0-7]*/i
BIN_NUMBER : /-?0b[0-1]*/i
FLOAT_NUMBER: /-?((\d+\.\d*|\.\d+|\d+)(e[-+]?\d+)?|\d+(e[-+]?\d+))/i
IMAG_NUMBER: (DEC_NUMBER | FLOAT_NUMBER) "j"i
COMPLEX_NUMBER: IMAG_NUMBER
              | "(" (FLOAT_NUMBER | DEC_NUMBER) /[+-]/ IMAG_NUMBER ")"
""",
)


class _PathTransformer(lark.Transformer):
  """Transforms a Lark parse-tree into a Path object."""

  @staticmethod
  def path(args: list[Any]) -> Path:
    return Path(*args)

  @staticmethod
  def identifier(args: list[lark.Token]) -> str:
    return str(args[0])

  @staticmethod
  def slice_key(args: list[str]) -> slice:
    sargs: list[Optional[int]] = [None, None, None]
    i = 0
    for a in args:
      if a == ":":
        i += 1
      else:
        sargs[i] = int(a) if a is not None else None
    return slice(*sargs)

  @staticmethod
  def number(args: list[str]) -> Union[int, float, complex]:
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
