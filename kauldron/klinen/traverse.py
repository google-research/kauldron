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

"""Module helper."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, Generic, Optional, TypeVar

from flax import linen as nn
from kauldron.klinen import module as module_lib

_T = TypeVar('_T')


class Future(Generic[_T]):
  """Value wrapper which is set later."""

  value: _T


def recursive_set_parent(module: _T) -> _T:
  """Reccursivelly assign the parent to the modules.

  Ater this function is called, all the `model.childs[0].child` will
  have their `_kd_name` (e.g. `'childs_3'`) and `_kd_future_parent` set.

  Args:
    module: The module to traverse.

  Returns:
    A copy of the module and sub-modules with parent set.
  """
  return recursive_replace(module, replace_fn=_replace_module)


def _replace_module(
    module: _T,
    *,
    attributes: dict[str, Any],
    name: Optional[str],
    future_parent: Future[module_lib.Module],
) -> _T:
  """Replace the module with updated name and parent fields."""
  new_module = dataclasses.replace(module, **attributes)
  assert new_module is not module
  new_module._kd_name = name  # pylint: disable=protected-access
  new_module._kd_future_parent = (  # pylint: disable=protected-access
      future_parent
  )
  return new_module


def recursive_replace(
    module: _T,
    *,
    name: Optional[str] = None,
    future_parent: Optional[Future[module_lib.Module]] = None,
    replace_fn: Callable[..., _T],
) -> _T:
  """Reccursivelly traverse the modules attribute and replace them.

  Args:
    module: The module to traverse.
    name: Module name
    future_parent: Module parent
    replace_fn: Function to replace the module with updated attributes.

  Returns:
    A copy of the module and sub-modules with parent set.
  """
  self_parent = Future()

  # Cache the modules shared across fields.
  # Note: This does not support circles (`module.child is module`)
  cache: dict[int, module_lib.Module] = {}

  # TODO(epot): Could have more optimized implementation (skip non-module
  # fields)
  attributes = {
      f.name: getattr(module, f.name)
      for f in dataclasses.fields(module)
      if f.init
  }
  module_attributes = {}
  for k, v in attributes.items():
    is_module_field = Future()
    is_module_field.value = False
    # `_map_over_modules_in_tree` implementation could likely be
    # optimized to be applied using list/dict comprehension.
    v = nn.module._map_over_modules_in_tree(  # pylint: disable=protected-access
        functools.partial(
            _set_module_name_and_parent,
            future_parent=self_parent,
            cache=cache,
            replace_fn=replace_fn,
            is_module_field=is_module_field,
        ),
        # Need to wrap inside dict, otherwise, flax do not infer attribute name.
        {k: v},
    )
    if is_module_field.value:
      module_attributes[k] = v[k]

  new_module = replace_fn(
      module,
      attributes=module_attributes,
      name=name,
      future_parent=future_parent,
  )

  self_parent.value = new_module
  return new_module


def _set_module_name_and_parent(
    prefix: str,
    leaf: _T,
    *,
    future_parent: Future[module_lib.Module],
    cache: dict[int, module_lib.Module],
    replace_fn: Callable[..., _T],
    is_module_field: Future[bool],
) -> _T:
  """Set the `_kd_name` and `_kd_future_parent` attribute."""
  if isinstance(leaf, module_lib.Module):
    is_module_field.value = True
    id_ = id(leaf)
    if id_ in cache:  # Already created
      return cache[id_]
    leaf = recursive_replace(
        leaf,
        name=prefix.removeprefix('_'),
        future_parent=future_parent,
        replace_fn=replace_fn,
    )
    cache[id_] = leaf
    return leaf
  else:
    return leaf
