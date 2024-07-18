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

"""Typing annotation utils."""

from __future__ import annotations

import difflib
import functools
import inspect
import os
import typing
from typing import Annotated, Any, Callable, Iterable, Optional, TypeVar

from etils import epy
import jax
from kauldron.kontext import paths
from kauldron.kontext import type_utils

_T = TypeVar("_T")

_key_token = object()
Key = Annotated[Any, _key_token]  # `str` or `path_builder_from()`
# `KeyTree` can wrap arbitrary PyTree objects when typing annotations cannot
# be infered (e.g `ray: KeyTree[v3d.Ray]`)
if typing.TYPE_CHECKING:
  KeyTree = Any  # TODO(b/254514368): Remove once pytype supports `Annotated`
else:
  KeyTree = Annotated[_T, _key_token]

REQUIRED = "__KEY_REQUIRED__"
_MISSING = object()

# Protocol to returns the Keys
_GET_KEY_PROTOCOL = "__kontext_keys__"

Tree = Any


def resolve_from_keyed_obj(
    context: Any,
    keyed_obj: Any,
    *,
    func: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
  """Resolve the Key annotations of an object for given context.

  Args:
    context: Any object from which the values are retrieved.
    keyed_obj: An instance of a class with fields annotated as Key, or
      implementing the `__kontext_keys__()` protocol.
    func: Optionally pass a function from which the signature should match the
      keys. This adds some extra checking to ensure the call function signature
      matches the provided keys.

  Returns:
    A dict mapping Key names to values from context corresponding to the paths
    determined by the Key fields.

  Raises:
    KeyError: If any non-optional keys are mapped to None.
  """
  try:
    key_paths = get_keypaths(keyed_obj)
    # Filter optional keys
    # If a key is `None`, we don't pass it to the function (instead rely on the
    # function argument default value).
    key_paths = {k: v for k, v in key_paths.items() if v is not None}

    _assert_no_required_keys(key_paths)
    if func is not None:
      _assert_signature_match(key_paths, func)

    return resolve_from_keypaths(context, key_paths)
  except Exception as e:  # pylint: disable=broad-exception-caught
    epy.reraise(e, f"Error for {type(keyed_obj).__qualname__}: ")


def resolve_from_keypaths(
    context: Any,
    key_paths: Tree[str],
) -> dict[str, Any]:
  """Get values for key_paths from context with useful errors when failing."""
  # There should not be any None values in the context.
  # No remaining None keys left. This constraint could be relaxed based
  # on use-case.
  _assert_no_none_keys(key_paths)
  key_values = jax.tree.map(
      lambda path: paths.get_by_path(context, path, default=_MISSING),
      key_paths,
  )
  _assert_no_missing_keys(context, key_paths, key_values)
  return key_values


def get_keypaths(keyed_obj: Any) -> dict[str, Tree[str] | None]:
  """Return a dictionary mapping Key-annotated fieldnames to their paths."""
  if hasattr(type(keyed_obj), _GET_KEY_PROTOCOL):
    return getattr(keyed_obj, _GET_KEY_PROTOCOL)()
  return {
      key: getattr(keyed_obj, key)
      for key in set(type_utils.get_annotated(keyed_obj, Key))
  }


def is_key_annotated(cls_or_obj: type[Any] | Any) -> bool:
  """Check if a given class or instance has fields annotated with `Key`."""
  # TODO(epot): `get_annotated` should recurse into `dict`,...
  return hasattr(cls_or_obj, _GET_KEY_PROTOCOL) or bool(
      type_utils.get_annotated(cls_or_obj, Key)
  )


def _assert_no_required_keys(key_paths: Tree[str]) -> None:
  """Raise an error if any of the key is `REQUIRED`."""
  missing_keys = [
      k for k, v in paths.flatten_with_path(key_paths).items() if v == REQUIRED
  ]
  if missing_keys:
    raise ValueError(
        f"Cannot resolve required keys: {missing_keys}.\n"
        "Keys should be specified during object construction."
    )


def _assert_no_none_keys(key_paths: Tree[str]) -> None:
  """Raise an error if any of the key is `REQUIRED`."""
  none_keys = [
      k for k, v in paths.flatten_with_path(key_paths).items() if v is None
  ]
  if none_keys:
    raise ValueError(
        f"Cannot resolve keys set to `None`: {none_keys}. Please open a bug"
        " with your use-case if you need this."
    )


def _assert_signature_match(
    key_paths: Tree[str],
    func: Callable[..., Any],
) -> None:
  """Validate that the signature of the function matches the key_paths."""
  sig = inspect.signature(func)
  try:
    sig.bind(**key_paths)  # Validate that the keys match the signature
  except TypeError as e:
    raise TypeError(
        f"Function {func.__qualname__} signature does not match the Key"
        " annotations of the object:\n"
        f" * Error: {e}\n"
        f" * Signature: {str(sig)}\n"
        f" * Provided (non-None) keys: {list(key_paths)}\n"
    ) from None


def _assert_no_missing_keys(context, key_paths, key_values) -> None:
  """Raise an error if a key cannot be found inside the context."""
  missing_values = {
      k for k, v in paths.flatten_with_path(key_values).items() if v is _MISSING
  }
  if missing_values:
    missing_keys = {
        k: str(v)
        for k, v in paths.flatten_with_path(key_paths).items()
        if k in missing_values
    }
    flat_paths = paths.flatten_with_path(context).keys()
    details = "\n".join(
        _get_missing_key_error_message(missing_key, path, flat_paths)
        for missing_key, path in missing_keys.items()
    )
    msg = _KeyErrorMessage(f"Invalid keys\n{details}")
    raise KeyError(msg)


def _get_missing_key_error_message(
    missing_key: str, missing_path: str, all_paths: Iterable[str]
) -> str:
  """Generate an error message with helpful suggestions for missing keypath."""
  suggestions = difflib.get_close_matches(
      missing_path, all_paths, n=3, cutoff=0
  )

  def common_prefix_length(x, path) -> int:
    return len(os.path.commonprefix((x, path)))

  sorted_by_common_prefix = sorted(
      all_paths,
      key=functools.partial(common_prefix_length, path=missing_path),
  )
  suggestions += sorted_by_common_prefix[-3:]
  suggestions = sorted(set(suggestions))
  return (
      f"Couldn't resolve path for {missing_key!r}:\n"
      + f"  {missing_path!r}\n"
      + "Did you mean one of:\n"
      + "\n".join([f"  {s!r}" for s in suggestions])
  )


class _KeyErrorMessage(str):
  """Used to support newlines in KeyError messages."""

  def __repr__(self):
    return str(self)
