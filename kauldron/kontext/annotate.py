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

"""Typing annotation utils."""
from __future__ import annotations

import difflib
import functools
import inspect
import os
from typing import Annotated, Any, Callable, Iterable, Optional

from kauldron.kontext import paths
from kauldron.kontext import type_utils

_key_token = object()
Key = Annotated[str, _key_token]

REQUIRED = "__KEY_REQUIRED__"

# Protocol to returns the Keys
_GET_KEY_PROTOCOL = "__kontext_keys__"


def get_from_keys_obj(
    tree: Any, keyed_obj: Any, *, func: Optional[Callable[..., Any]] = None
) -> dict[str, Any]:
  """Resolve the Key annotations of an object for given context.

  Args:
    tree: Any object from which the values are retrieved.
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
  key_paths = _get_keypaths(keyed_obj)
  missing_keys = [k for k, v in key_paths.items() if v == REQUIRED]
  if missing_keys:
    raise ValueError(
        f"Cannot resolve required keys: {missing_keys} for {keyed_obj}.\n"
        "Keys should be defined during object construction."
    )
  optional_keys = {  # treat Optional[Key] as optional only if set to None
      k
      for k in type_utils.get_optional_fields(keyed_obj)
      if k in key_paths and key_paths[k] is None
  }

  if func is not None:
    # If a function is passed (and doesn't accept **kwargs), then ensure that
    # it has a parameter for each key, and that the parameter for each optional
    # key also specifies a default value
    parameters = inspect.signature(func).parameters
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if not has_var_kwargs:
      missing_params = {k for k in key_paths if k not in parameters}
      if missing_params:
        raise TypeError(
            f"Function {func.__name__} is missing parameters for key(s) "
            f"{missing_params}. Expected {key_paths.keys()} but got "
            f"{parameters}."
        )
      args_with_default = {
          name
          for name, param in parameters.items()
          if param.default != inspect.Parameter.empty
      }
      missing_default = {k for k in optional_keys if k not in args_with_default}
      if missing_default:
        raise TypeError(
            f"Key(s) {missing_default!r} are marked as optional, so the "
            f"corresponding argument in {func.__name__} has to specify "
            "default value."
        )

  return _get_values_from_context(tree, key_paths, optional_keys)


def _get_values_from_context(
    context: Any, key_paths: dict[str, str], optional_keys=Optional[set[str]]
) -> dict[str, Any]:
  """Get values for key_paths from context with useful errors when failing."""
  optional_keys = set() if optional_keys is None else optional_keys
  key_values = {
      key: paths.get_by_path(context, path) for key, path in key_paths.items()
  }
  missing_keys = {
      key: key_paths[key]
      for key, value in key_values.items()
      if value is None and key not in optional_keys
  }
  if missing_keys:
    flat_paths = paths.flatten_with_path(context).keys()
    details = "\n".join(
        _get_missing_key_error_message(missing_key, path, flat_paths)
        for missing_key, path in missing_keys.items()
    )
    msg = _KeyErrorMessage(f"Missing keys\n{details}")
    raise KeyError(msg)
  return key_values


def _get_missing_key_error_message(
    missing_key: str, missing_path: str, all_paths: Iterable[str]
) -> str:
  """Generate an error message with helpful suggestions for missing keypath."""
  suggestions = difflib.get_close_matches(
      missing_path, all_paths, n=3, cutoff=0
  )

  def common_prefix_length(x, path):
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


def _get_keypaths(keyed_obj: Any) -> dict[str, str]:
  """Return a dictionary mapping Key-annotated fieldnames to their paths."""
  if hasattr(type(keyed_obj), _GET_KEY_PROTOCOL):
    return getattr(keyed_obj, _GET_KEY_PROTOCOL)()
  return {
      key: getattr(keyed_obj, key)
      for key in set(type_utils.get_annotated(keyed_obj, Key))
  }


def is_key_annotated(cls_or_obj: type[Any] | Any) -> bool:
  """Check if a given class or instance has fields annotated with `Key`."""
  return hasattr(cls_or_obj, _GET_KEY_PROTOCOL) or bool(
      type_utils.get_annotated(cls_or_obj, Key)
  )


class _KeyErrorMessage(str):
  """Used to support newlines in KeyError messages."""

  def __repr__(self):
    return str(self)
