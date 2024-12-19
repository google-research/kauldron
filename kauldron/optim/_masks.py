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

"""Masks utils."""

from collections.abc import Callable, Sequence
import re
from typing import Any

import jax

_PyTree = Any


# Improvements:
# * Could add `exclude=` kwargs, similar to `glob()`.


def select(pattern: str | Sequence[str]) -> Callable[[_PyTree], _PyTree]:
  r"""Create a mask which selects only the sub-pytree matching the pattern.

  * `xx` will match all `{'xx': ...}` dict anywhere inside the tree. Note that
    the match is strict, so `xx` will NOT match `{'xxyy': }`
  * `xx.yy` will match `{'xx': {'yy': ...}}` dict
  * Regex are supported, when using regex, make sure to escape `.` (e.g.
    `xx\.yy[0-9]+`)

  Example:

  ```python
  mask_fn = kg.optim.select("lora")

  mask_fn({
      'layer0': {
          'lora': {
              'a': jnp.zeros(),
              'b': jnp.zeros(),
          },
          'weights': jnp.zeros(),
          'bias': jnp.zeros(),
      }
  }) == {
      'layer0': {
          'lora': {
              'a': True,
              'b': True,
          },
          'weights': False,
          'bias': False,
      }
  }
  ```

  Args:
    pattern: The pattern to include. Everything else will be `False`.

  Returns:
    The optax mask factory.
  """

  # Convert the pattern to a regex.
  if isinstance(pattern, str):
    pattern = [pattern]

  pattern_regexes = [_make_regex(p) for p in pattern]

  def _path_match_pattern(path: jax.tree_util.KeyPath) -> bool:
    path_str = ".".join(_jax_key_entry_to_str(p) for p in path)
    return any(bool(p.search(path_str)) for p in pattern_regexes)

  def _make_mask(tree: _PyTree) -> _PyTree:
    # TODO(epot): Replace by `jax.tree.flatten_with_path` once Colab is updated
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(tree)

    # Parse each leaves
    leaves = []
    for path, _ in leaves_with_path:
      leaves.append(_path_match_pattern(path))

    # Restore the tree structure.
    return jax.tree.unflatten(treedef, leaves)

  return _make_mask


def exclude(pattern: str | Sequence[str]) -> Callable[[_PyTree], _PyTree]:
  """Create a mask which selects all nodes except the ones matching the pattern.

  This is the inverse of `select()`. All the tree nodes will be `True`, except
  the ones matching the pattern.

  Args:
    pattern: The pattern to exclude. See `select()` for more details.

  Returns:
    The optax mask factory.
  """
  make_select_mask = select(pattern)

  def _make_mask(tree: _PyTree) -> _PyTree:
    # Invert the select mask.
    tree = make_select_mask(tree)
    return jax.tree.map(lambda x: not x, tree)

  return _make_mask


_REGEX_SPECIAL_CHARS = set("()[]?+*^$|\\")


def _make_regex(pattern: str) -> re.Pattern[str]:
  # Auto-detect regex and forward them as-is.
  if any(c in _REGEX_SPECIAL_CHARS for c in pattern):
    pass
  else:  # Otherwise, escape special characters (`.`).
    pattern = re.escape(pattern)

  pattern = rf"(?:^|\.){pattern}(?:$|\.)"
  return re.compile(pattern)


def _jax_key_entry_to_str(
    jax_key_entry: jax.tree_util.KeyEntry,
) -> str:
  """Convert a JaxKeyEntry into a valid `kontext.Path` element."""
  match jax_key_entry:
    case jax.tree_util.DictKey(key):
      return key
    case _:
      raise TypeError(f"Unknown key entry type {type(jax_key_entry)}")
