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

"""Utils."""

from collections.abc import Sequence
from typing import Any

from etils import epy
from kauldron.kontext import glob_paths
from kauldron.kontext import path_parser
from kauldron.kontext import paths

_MISSING = object()


def filter_by_path(
    obj: paths.Context,
    path: str | tuple[str, ...] | paths.AbstractPath,
) -> paths.Context:
  """Filters a context by a path.

  This function returns a subset of the context. This only supports `dict`,
  `list` and `tuple` types. Non-matching list items are removed.

  Example:

  ```python
  assert kontext.filter_by_path(
      {
          'a': {
              'b': {
                  'c': 1,
                  'c1': 1,
                  'c2': 1,
              },
              'b2': {
                  'c': {'d': 1},
                  'c1': {'d': 1},
                  'c2': {'d': 1},
              },
          },
      },
      '**.c',  # Select only the `c` keys.
  ) == {
      'a': {
          'b': {
              'c': 1,
          },
          'b2': {
              'c': {'d': 1},
          },
      },
  }
  ```

  Args:
    obj: The context to filter.
    path: The path to filter by.

  Returns:
    A subset of the context.
  """
  match path:
    case str():
      path = glob_paths.GlobPath.from_str(
          path
      )  # Otherwise, try parsing key as path.
    case tuple() as parts:
      path = glob_paths.GlobPath(*parts)
    case paths.AbstractPath():
      path = glob_paths.GlobPath(*path.parts)
    case _:
      raise TypeError(f"Unknown key/path {path} of type{type(path)}")

  try:
    return _filter_by_path(obj, path.parts)
  except Exception as e:  # pylint: disable=broad-exception-caught
    epy.reraise(e, f"Error extracting {str(path)!r}:\n")


def _filter_by_path(
    obj: paths.Context,
    parts: Sequence[paths.Part],
    *,
    missing_ok: bool = False,
    merge_with: dict[Any, Any] | None = None,
) -> paths.Context:
  """Filter implementation."""
  if not parts:
    return obj

  wrapper = glob_paths.Node.make(obj)
  part, *rest = parts

  # During glob, the object might contains branch which do not match
  if missing_ok and part not in wrapper:
    if merge_with:
      return wrapper.from_items(merge_with)
    else:
      return _MISSING  # Leaf not found, do not assign this branch

  if part == path_parser.Wildcard.DOUBLE_STAR:
    if isinstance(wrapper, glob_paths.Leaf):  # Leaf, do not recurse
      return obj if not rest else _MISSING
    else:
      # Recurse on each sub-nodes
      child_values = {}
      for k, v in wrapper.get_items(path_parser.Wildcard.STAR):
        with epy.maybe_reraise(lambda: f"- In ** = {k!r}\n"):  # pylint: disable=cell-var-from-loop
          # Propagate the `**` to each items
          filtered_v = _filter_by_path(v, parts)
        if filtered_v is not _MISSING:  # Only add the leafs which matches
          child_values[k] = filtered_v

      # Try to assign the rest in the current context
      return _filter_by_path(
          obj,
          rest,
          missing_ok=True,
          merge_with=child_values,
      )
  else:
    values = {}
    for k, v in wrapper.get_items(part):
      with epy.maybe_reraise(lambda: f"- In {k!r}\n"):  # pylint: disable=cell-var-from-loop
        filtered_v = _filter_by_path(v, rest)
      if filtered_v is not _MISSING:  # Only add the leafs which matches
        values[k] = filtered_v

    if merge_with:
      # TODO(epot): There's edge cases if merge_with and values conflict,
      # like: `**.b.a` for `{'b': {'a': 1, 'b': {'a': 1}}}`
      values = merge_with | values
    return wrapper.from_items(values)
