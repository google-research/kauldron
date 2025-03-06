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

import collections
from collections.abc import Mapping, Sequence
from typing import Any, Callable, ClassVar, Self, TypeVar, Union, overload

from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import jax.tree_util
from kauldron.kontext import path_parser
import ml_collections

_T = TypeVar("_T")
PyTree = Union[_T, Sequence["PyTree[_T]"], Mapping[str, "PyTree[_T]"], Any]

# Context object is a nested structure (dict, dataclass)
Context = Any
Part = path_parser.Part


JaxKeyEntry = Union[
    jax.tree_util.SequenceKey,
    jax.tree_util.DictKey,
    jax.tree_util.GetAttrKey,
    jax.tree_util.FlattenedIndexKey,
]


def _is_valid_part(part: Any, *, wildcard_ok: bool = False) -> bool:
  if isinstance(part, tuple):
    return all(_is_valid_part(p, wildcard_ok=wildcard_ok) for p in part)
  elif not wildcard_ok and isinstance(part, path_parser.Wildcard):
    return False  # Wildcards are not supported
  else:
    return isinstance(part, Part)


class AbstractPath(collections.abc.Sequence):
  """Represents a string path."""

  __slots__ = ("parts",)

  _SUPPORT_GLOB: ClassVar[bool]

  def __init__(self, *parts: Part):
    if not _is_valid_part(parts, wildcard_ok=self._SUPPORT_GLOB):
      raise ValueError(f"Invalid part(s) {parts}")
    self.parts: tuple[Part, ...] = parts

  @overload
  def __getitem__(self, key: int) -> Part:
    ...

  @overload
  def __getitem__(self, key: slice) -> Self:
    ...

  def __getitem__(self, key: int | slice) -> Part | Self:
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
    if isinstance(other, type(self)):
      return self.parts == other.parts
    else:
      return False

  def __repr__(self) -> str:
    r = "".join(_format_part(part) for part in self.parts)
    r = r.removeprefix(".")
    return r

  @classmethod
  def from_str(cls, str_path: str) -> Self:
    return cls(*path_parser.parse_parts(str_path))

  @classmethod
  def from_jax_path(cls, jax_path: tuple[JaxKeyEntry, ...]) -> Self:
    """Create Path object from a jax path.

    Args:
      jax_path: A jax key path for example from
        `jax.tree_util.tree_flatten_with_path(...)`.

    Returns:
      The corresponding `kontext.Path`.
    """
    return cls(*(_jax_key_entry_to_kd_path_element(p) for p in jax_path))

  def set_in(self, context: Context, value: Any) -> None:
    raise NotImplementedError("Abstract method")

  def relative_to(self, other: AbstractPath) -> Self:
    if len(self.parts) < len(other.parts):
      raise ValueError(f"{self} is not a subpath of {other}")
    common_parts = self.parts[: len(other.parts)]
    if common_parts != other.parts:
      raise ValueError(f"{self} is not a subpath of {other}")
    return type(self)(*self.parts[len(other.parts) :])


class Path(AbstractPath):
  """Represents a (non-glob) string path."""

  _SUPPORT_GLOB = False

  # TODO(klausg): docstring, annotations and better name
  def get_from(
      self,
      context: Context,
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
        if isinstance(part, str) and hasattr(result, part):
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
            # TODO(epot): Better error message for dict (use did you mean ?)
            raise KeyError(
                f"Could not find path '{self}' in the"
                f" {type(context).__name__} object. The"
                f" {type(result).__name__} has no attributes/key:"
                f" {part!r}.{struct}"
            ) from None
          return default
    return result

  def set_in(self, context: Context, value: Any) -> None:
    """Set the object in the path."""
    root = context

    *parts, target = self.parts
    for part in parts:
      root = root[part]
      if not isinstance(root, (list, dict, ml_collections.ConfigDict)):
        raise TypeError(
            f"Cannot overwrite value {self}: {part} is unsuported type"
            f" {type(root)}. Please open an issue if this should be fixed."
        )

    root[target] = value


def _jax_key_entry_to_kd_path_element(
    jax_key_entry: JaxKeyEntry,
) -> Part:
  """Convert a JaxKeyEntry into a valid `kontext.Path` element."""
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


def get_by_path(
    obj: Context,
    path: str | tuple[str, ...] | Path,
    default=...,
) -> Any:
  """Get (nested) item or attribute by given path.

  Args:
    obj: The object / dict / dataclass to retrieve the value from.
    path: The path to retrieve, either as a string "foo[1]", a tuple ("foo", 1)
      or a Path object.
    default: return value if no value is found. By default, raise an error.

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
      return None if default is ... else default
    case _:
      raise TypeError(f"Unknown key/path {path} of type{type(path)}")

  return path.get_from(obj, default=default)


def flatten_with_path(
    pytree: PyTree[_T],
    *,
    prefix: str = "",
    separator: str | None = None,
    is_leaf: Callable[[Any], bool] | None = None,
) -> dict[str, _T]:
  """Flatten any PyTree / ConfigDict into a dict with 'keys.like[0].this'."""
  if isinstance(pytree, ml_collections.ConfigDict):
    # Normalize ConfigDict to dict
    pytree = pytree.to_dict()
  flat_tree_items, _ = jax.tree_util.tree_flatten_with_path(
      pytree, is_leaf=is_leaf
  )
  prefix = (jax.tree_util.GetAttrKey(prefix),) if prefix else ()

  def _format_path(jax_path):
    path = Path.from_jax_path(prefix + jax_path)
    if separator is None:
      return str(path)
    else:
      return separator.join(str(p) for p in path.parts)

  return {_format_path(jax_path): value for jax_path, value in flat_tree_items}


def _format_part(part: Any) -> str:
  """Format a single part of a path."""
  if isinstance(part, str) and part.isidentifier():
    return "." + part
  elif isinstance(part, path_parser.Wildcard):
    return "." + str(part)
  elif isinstance(part, slice):
    return f"[{_format_slice(part)}]"
  elif part == ...:
    return "..."
  elif isinstance(part, tuple):
    parts_str = ",".join(_format_axis(ax) for ax in part)
    return f"[{parts_str}]"
  else:
    # TODO(epot): Should check all cases and end with `else: raise`
    return f"[{part!r}]"


def _format_axis(axis: Any) -> str:
  """Format a single axis of a tensor slice."""
  if isinstance(axis, int):
    return str(axis)
  elif isinstance(axis, slice):
    return _format_slice(axis)
  elif axis == ...:
    return "..."
  elif axis is None:
    return "None"
  else:
    raise ValueError(f"Unknown axis {axis} of type {type(axis)}")


def _format_slice(s: slice) -> str:
  """Format a silce object into a colon-separated string like '::2'."""
  fm = [s.start, ":", s.stop]
  if s.step is not None:
    fm += [":", s.step]
  return "".join(str(f) for f in fm if f is not None)
