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

"""Paths."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence, MutableMapping
import dataclasses
from typing import Any

from etils import epy
from kauldron.kontext import path_parser
from kauldron.kontext import paths

# Context object is a nested structure (dict, dataclass)
Context = Any


def set_by_path(
    obj: Context,
    path: str | tuple[str, ...] | paths.AbstractPath,
    value: Any,
):
  """Mutate the `obj` to set the value."""
  match path:
    case str():
      path = GlobPath.from_str(path)  # Otherwise, try parsing key as path.
    case tuple() as parts:
      path = GlobPath(*parts)
    case paths.AbstractPath():
      pass
    case _:
      raise TypeError(f"Unknown key/path {path} of type{type(path)}")

  return path.set_in(obj, value)


class GlobPath(paths.AbstractPath):
  """Represents a string path."""

  _SUPPORT_GLOB = True

  def set_in(self, context: Context, value: Any) -> None:
    """Set the object in the path."""
    try:
      _set_in(context, self.parts, value)
    except Exception as e:  # pylint: disable=broad-exception-caught
      epy.reraise(e, prefix=f"Error trying to mutate path {self}: ")


@dataclasses.dataclass(frozen=True)
class _Node(abc.ABC):
  """Helper to help the recursion.

  Wrap a context object, and provides methods to access/mutate the object.
  This unify the access to all tree objects (dict, list, ...).
  """

  obj: Any

  @classmethod
  def make(cls, obj) -> _Node:
    match obj:
      case MutableMapping():
        return _Dict(obj)
      case list():
        return _List(obj)
      case _:
        return _Leaf(obj)

  def get_items(self, part: path_parser.Part) -> Iterable[Any]:
    if part == path_parser.Wildcard.STAR:
      return self.values()
    elif part == path_parser.Wildcard.DOUBLE_STAR:
      raise RuntimeError("Should not get `**`")
    else:
      try:
        return (self[part],)
      except Exception as e:  # pylint: disable=broad-exception-caught
        epy.reraise(e, prefix=f"Error accessing {part!r} in {type(self.obj)}: ")

  @abc.abstractmethod
  def __setitem__(self, part: path_parser.Part, value: Any) -> None:
    raise NotImplementedError

  @abc.abstractmethod
  def __getitem__(self, part: path_parser.Part) -> Any:
    raise NotImplementedError

  @abc.abstractmethod
  def __contains__(self, part: path_parser.Part) -> bool:
    raise NotImplementedError

  @abc.abstractmethod
  def values(self) -> Iterable[Any]:
    raise NotImplementedError


class _Dict(_Node):
  """Dict or mapping node."""

  def __setitem__(self, part: path_parser.Part, value: Any) -> None:
    # Note: The key do not need to exists. It can be created.
    self.obj[part] = value

  def __getitem__(self, part: path_parser.Part) -> Any:
    # TODO(epot): Validate key
    return self.obj[part]

  def __contains__(self, part: path_parser.Part) -> bool:
    # TODO(epot): Validate key
    return part in self.obj

  def values(self) -> Iterable[Any]:
    yield from self.obj.values()


class _List(_Node):
  """List node."""

  def __setitem__(self, part: path_parser.Part, value: Any) -> None:
    if isinstance(part, slice):
      raise NotImplementedError(
          "Slice keys are not yet supported. Please open an issue."
      )
    if not isinstance(part, int):
      raise TypeError(f"Expected list index, got {part!r}.")
    if abs(part) >= len(self.obj):
      raise KeyError(f"Key {part} out of range (length {len(self.obj)})")
    self.obj[part] = value

  def __getitem__(self, part: path_parser.Part) -> Any:
    # TODO(epot): Validate key
    assert not isinstance(part, path_parser.Wildcard)
    return self.obj[part]

  def __contains__(self, part: path_parser.Part) -> bool:
    # TODO(epot): Validate key
    assert not isinstance(part, path_parser.Wildcard)
    if isinstance(part, int) and abs(part) < len(self.obj):  # part present
      return True
    # TODO(epot): Support slices
    return False

  def values(self) -> Iterable[Any]:
    yield from self.obj


# TODO(epot): Add tuple which are immutable, but can still be recursed into
# class _Tuple(_List):


class _Leaf(_Node):
  """Leaf node."""

  def __setitem__(self, key: str, value: Any) -> None:
    del value
    raise ValueError(
        f"Cannot mutate leaf {type(self.obj)}. Tried to overwrite key"
        f" {key!r}. Only dict, list can be recursed into."
    )

  def __getitem__(self, part: path_parser.Part) -> Any:
    raise RuntimeError  # Should never be called

  def values(self) -> Iterable[Any]:
    raise ValueError(
        f"Cannot recurse inside {type(self.obj)} (not a dict or list)"
    )

  def __contains__(self, part: path_parser.Part) -> bool:
    # Leafs can never be recursed into
    return False


def _set_in(
    context: Context,
    parts: Sequence[paths.Part],
    value: Any,
    *,
    missing_ok: bool = False,
) -> bool:
  """Recursively set the value from the path."""
  # Field reference are resolved in the config.
  if not parts:
    raise ValueError("Path is empty")

  wrapper = _Node.make(context)
  part, *rest = parts

  # During glob, the object might contains branch which do not match
  if missing_ok and part not in wrapper:
    return  # Leaf not found, do not assign this branch

  if not rest:  # Nothing left to recurse on, assign the leaf.
    if isinstance(part, path_parser.Wildcard):
      # I don't think there's a use-case for this. For glob, this would create
      # ambiguity too.
      raise ValueError("Wildcards cannot be located at the end of a path.")
    wrapper[part] = value
  elif part == path_parser.Wildcard.DOUBLE_STAR:
    # Try to assign the rest in the current context
    _set_in(context, rest, value, missing_ok=True)
    if isinstance(wrapper, _Leaf):  # Leaf, do not recurse
      return
    # Recurse over all elements
    for new_context in wrapper.get_items(path_parser.Wildcard.STAR):
      _set_in(new_context, parts, value)  # Propagate the `**` to the leaves
  else:  # Otherwise, recurse.
    for new_context in wrapper.get_items(part):
      _set_in(new_context, rest, value)
  # TODO(epot): Reraise with the full path branch in which the error occured
