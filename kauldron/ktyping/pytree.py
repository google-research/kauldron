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

"""Basic PyTree type definition."""

from __future__ import annotations

from typing import Any

import jax
from kauldron.ktyping import internal_typing
from kauldron.ktyping import utils

Missing = internal_typing.Missing
MISSING = internal_typing.MISSING


# MARK: PyTreeMeta
class _PyTreeMeta(type):
  """Metaclass for creating pytree types."""

  leaf_type: Any | Missing
  structure_spec: str | Missing

  def __new__(
      mcs,
      name: str,
      *,
      leaf_type: type[Any] | Missing = MISSING,
      structure_spec: str | Missing = MISSING,
  ):
    return super().__new__(
        mcs,
        name,
        (),
        {"leaf_type": leaf_type, "structure_spec": structure_spec},
    )

  def __init__(cls, *args, **kwargs):
    del args, kwargs  # unused
    super().__init__(cls)

  def __getitem__(
      cls,
      args: type[Any] | tuple[type[Any], str],  # PyTree[T]  | PyTree[T, "S"]
  ) -> _PyTreeMeta:
    """Item access syntax is used to create new pytree types."""
    structure_spec = MISSING
    if not isinstance(args, tuple):
      leaf_type = args
    elif len(args) == 2:
      leaf_type, structure_spec = args
      structure_spec = _validate_structure_spec(structure_spec)
    else:
      raise TypeError(f"Expected 1 or 2 arguments, got {len(args)}")

    leaf_type_name = utils.get_type_name(leaf_type)

    if cls.leaf_type is not MISSING:
      raise TypeError(
          f"Trying to redefine leaf_type of {cls.__name__} which is"
          f" {utils.get_type_name(cls.leaf_type)} with"
          f" {leaf_type}={leaf_type_name}."
      )

    if structure_spec is MISSING:
      name = f"{cls.__name__}[{leaf_type_name}]"
    else:
      name = f"{cls.__name__}[{leaf_type_name}, {structure_spec!r}]"

    return _PyTreeMeta(
        name=name,
        leaf_type=leaf_type,
        structure_spec=structure_spec,
    )

  def __instancecheck__(cls, instance: Any) -> bool:
    return jax.tree.reduce(
        lambda prev, x: prev and isinstance(x, cls.leaf_type), instance, True
    )

  def __repr__(cls):
    return cls.__name__

  def __call__(cls, *args, **kwargs):
    """Raises a RuntimeError to prevent accidental PyTree(T) syntax."""
    args_str = ", ".join(repr(a) for a in args)
    raise RuntimeError(
        f"{cls.__name__} cannot be instantiated. Did you mean to write"
        f" {cls.__name__}[{args_str}]?"
    )


def _validate_structure_spec(name: Any) -> str:
  """Validates and normalizes a structure spec string."""
  if not isinstance(name, str):
    raise TypeError(
        f"Structure name must be a string, got {type(name).__name__}."
    )
  name = name.strip()
  if not name.startswith(internal_typing.STRUCTURE_KEY_PREFIX):
    raise TypeError(
        "Structure name must start with"
        f" '{internal_typing.STRUCTURE_KEY_PREFIX}', got {name!r}."
    )
  if len(name) < 2:
    raise TypeError(
        "Structure name must have at least one character after"
        f" '{internal_typing.STRUCTURE_KEY_PREFIX}', got {name!r}."
    )
  return name


PyTree = _PyTreeMeta("PyTree")  # pylint: disable=invalid-name


# MARK: Utils


def _jax_key_entry_to_str(
    jax_key_entry: Any,
) -> str:
  """Convert a JaxKeyEntry into a str representation."""
  # pytype: disable=match-error
  match jax_key_entry:
    case jax.tree_util.GetAttrKey(name):
      return name
    case jax.tree_util.DictKey(key):
      return f"[{key!r}]"
    case jax.tree_util.SequenceKey(idx):
      return f"[{idx!r}]"
    case jax.tree_util.FlattenedIndexKey(key):
      return f"[{key!r}]"
  # pytype: enable=match-error
  raise TypeError(f"Unknown key entry type {type(jax_key_entry)}")


def jax_path_to_str(jax_path: tuple[Any, ...]) -> str:
  """Convert a JaxPath into a str representation."""
  return "".join(_jax_key_entry_to_str(key) for key in jax_path)


def is_pytree_type(type_: Any) -> bool:
  """Returns True if the type is a PyTree type."""
  return isinstance(type_, _PyTreeMeta)
