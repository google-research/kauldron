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
import inspect
from typing import Any, Optional, Callable

from kauldron.core import paths
import kauldron.typing as ktyping
from kauldron.utils import type_utils


def resolve_kwargs(
    keyed_obj: Any, context: Any, func: Optional[Callable[..., Any]] = None
) -> dict[str, Any]:
  """Resolve the Key annotations of an object for given context.

  Args:
    keyed_obj: An instance of a class with fields annotated as Key.
    context: Any object from which the values are retrieved.
    func: Optionally pass a function to be called with the kwargs. This
      restricts the gathered kwargs to the arguments of that function and treats
      arguments with default parameters as optional.

  Returns:
    A dict mapping Key names to values from context corresponding to the paths
    determined by the Key fields.

  Raises:
    KeyError: If any non-optional keys are mapped to None.
  """
  key_paths = get_keypaths(keyed_obj)
  optional_keys = {  # treat Optional[Key] as optional only if set to None
      k
      for k in type_utils.get_optional_fields(keyed_obj)
      if k in key_paths and key_paths[k] is None
  }

  if func is not None:
    # If a function is passed then only resolve values for its arguments
    # (unless it accepts a **kwargs parameter)
    parameters = inspect.signature(func).parameters
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if not has_var_kwargs:
      key_paths = {k: v for k, v in key_paths.items() if k in parameters}
    # treat arguments with defaults as optional
    args_with_default = {
        name
        for name, param in parameters.items()
        if param.default != inspect.Parameter.empty
    }
    optional_keys |= args_with_default

  return get_values_from_context(context, key_paths, optional_keys)


def get_values_from_context(
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
    flat_keys = paths.tree_flatten_with_path(context).keys()
    suggestions = []
    details = ""
    for key, path in missing_keys.items():
      suggestion = difflib.get_close_matches(key, flat_keys, n=1, cutoff=0)
      if suggestion:
        suggestions.append(
            f"{key}:\n  Not found:    {path!r}\n  Did you mean:"
            f" {suggestion[0]!r} ?"
        )
      else:
        suggestions.append(
            f"{key}:\n  Not found:    {path!r}\n. Context keys: {flat_keys} "
        )
      details = "\n".join(suggestions)

    msg = _KeyErrorMessage(f"Missing keys:\n{details}")
    raise KeyError(msg)
  return key_values


def get_keypaths(keyed_obj: Any) -> dict[str, str]:
  """Return a dictionary mapping Key-annotated fieldnames to their paths."""
  return {
      key: getattr(keyed_obj, key)
      for key in set(type_utils.get_annotated(keyed_obj, ktyping.Key))
  }


def is_key_annotated(cls_or_obj: type[Any] | Any) -> bool:
  """Check if a given class or instance has fields annotated with `Key`."""
  return bool(type_utils.get_annotated(cls_or_obj, ktyping.Key))


class _KeyErrorMessage(str):
  """Used to support newlines in KeyError messages."""

  def __repr__(self):
    return str(self)
