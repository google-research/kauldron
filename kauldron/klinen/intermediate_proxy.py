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

"""Intermediate proxy output."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional, TypeAlias, TypeVar

from etils import epy
import flax
import jax
from kauldron.klinen import intermediate
from kauldron.klinen import module as module_lib
from kauldron.klinen import traverse

_FnT = TypeVar('_FnT')
_T = TypeVar('_T')


# TODO(epot): Better support for `nn.compact` (`inter['Dense_0'].tmp`)
# TODO(epot): Fix non-tree modules (e.g. `Sequential([nn.relu])`)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _ModuleProxy:
  """Proxy of a single module exposing the captured intermediates."""

  _cls_name: str
  # Fields: `xxx: Intermediate[T]`
  _intermediate_fields: dict[str, intermediate.IntermediateDescriptor]
  # Fields: Nested modules (`xxx: nn.Module`)
  _module_attributes: dict[str, Any]
  # Flax name and parent
  _name: str
  _future_parent: traverse.Future[Optional[_ModuleProxy]]

  # Intermediate values
  # Accessible as `model.intermediate_val`
  _intermediate_attributes: dict[str, Any] = dataclasses.field(
      default_factory=dict
  )
  # Accessible as `model['intermediate_val']`
  _intermediate_items: dict[str, Any] = dataclasses.field(default_factory=dict)

  @classmethod
  def from_module(
      cls,
      module: module_lib.Module,
      *,
      attributes: dict[str, Any],
      name: Optional[str],
      future_parent: traverse.Future[Optional[_ModuleProxy]],
      cache: _Cache,
  ) -> _ModuleProxy:
    """Create the proxy."""
    assert id(module) not in cache
    proxy = cls(
        _cls_name=type(module).__name__,
        _intermediate_fields=_get_intermediate_fields(module),
        _module_attributes=attributes,
        _name=name,
        _future_parent=future_parent,
    )
    cache[id(module)] = proxy
    return proxy

  def __getattr__(self, name: str) -> Any:
    # Accessing a child module
    if name in self._module_attributes:
      return self._module_attributes[name]

    if name not in self._intermediate_fields:
      raise AttributeError(
          f'No attribute {self._cls_name}.{name}. Only `Intermediate[]` and'
          ' sub-module attributes are available.'
      )
    # Accessing an intermediate value
    return self._intermediate_attributes[name]

  def __getitem__(self, key: str) -> Any:
    return self._intermediate_items[key]

  @property
  def _attribute_names(self) -> list[str]:
    """List of defined attribute names."""
    return list(self._module_attributes) + list(self._intermediate_attributes)

  def __repr__(self) -> str:
    content = {k: getattr(self, k) for k in self._attribute_names}
    # extra_values are the `.sow` and uncaptured values.
    if self._intermediate_items:
      content['__getitem__'] = self._intermediate_items
    return epy.Lines.make_block(
        self._cls_name,
        content=content,
    )

  def __dir__(self) -> list[str]:
    """Available attributes."""
    return self._attribute_names

  @property
  def _parent(self) -> Optional[_ModuleProxy]:
    """Returns the parent."""
    if self._future_parent is not None:
      return self._future_parent.value
    return None

  @property
  def _parent_names(self) -> list[str]:
    """List of path (excluding the first)."""

    parent_names = []
    parent = self
    while parent._parent is not None:  # pylint: disable=protected-access  # pytype: disable=attribute-error
      parent_names.append(parent._name)  # pylint: disable=protected-access
      parent = parent._parent  # pylint: disable=protected-access

    return list(reversed(parent_names))

  def _set_intermediate(self, intermediate_dict: dict[str, Any]) -> None:
    """Assign the intermediate values to `self`."""
    # Get the inner-most dict
    values = intermediate_dict
    for name in self._parent_names:
      values = values.get(name, {})

    # Pop all intermediates
    for name, descriptor in self._intermediate_fields.items():
      # Intermediate was set, pop
      if descriptor._collection_name in values:  # pylint: disable=protected-access
        value = values.pop(descriptor._collection_name)  # pylint: disable=protected-access
      # Intermediate not set, but default exists
      elif descriptor.field.default is not dataclasses.MISSING:
        value = descriptor._default  # pylint: disable=protected-access
      # Internediate not set and missing
      else:
        continue
      self._intermediate_attributes[name] = value

    # Eventually other intermediates (from `.sow()`)
    self._intermediate_items = dict(values)
    values.clear()

    # Eventually clear the dict
    all_dicts = {}
    values = intermediate_dict
    for name in self._parent_names:
      all_dicts[name] = values
      values = values.get(name, {})

    for name, dict_ in reversed(all_dicts.items()):
      if not dict_.get(name, None):  # Empty dict, pop
        dict_.pop(name, None)

  def tree_flatten(
      self,
  ) -> tuple[list[Any], _ModuleProxy]:
    """`jax.tree_utils` support."""
    flat_values = [
        self._module_attributes,
        self._intermediate_attributes,
        self._intermediate_items,
    ]
    return (flat_values, self)  # pytype: disable=bad-return-type

  @classmethod
  def tree_unflatten(
      cls,
      metadata: _ModuleProxy,
      array_field_values: list[Any],
  ) -> _ModuleProxy:
    """`jax.tree_utils` support."""
    [
        module_attributes,
        intermediate_attributes,
        intermediate_items,
    ] = array_field_values
    return dataclasses.replace(
        metadata,
        _module_attributes=module_attributes,
        _intermediate_attributes=intermediate_attributes,
        _intermediate_items=intermediate_items,
    )


def _get_intermediate_fields(
    module: module_lib.Module,
) -> dict[str, intermediate.IntermediateDescriptor]:
  """Extract only `Intermediate` fields."""
  intermediate_fields = {}
  for cls in type(module).mro():
    for name, value in cls.__dict__.items():
      if isinstance(value, intermediate.IntermediateDescriptor):
        intermediate_fields[name] = value
  return intermediate_fields


@jax.tree_util.register_pytree_node_class
class ModuleIntermediateProxy:
  """Module-like object which contain the intermediate values."""

  _proxy: _ModuleProxy

  def __init__(self, module: module_lib.Module):
    # While `_finalized` is `False`, the proxy cannot be used by the user.
    self._finalized: bool = False
    self._module = module
    self._intermediate_dict: Optional[flax.core.scope.FrozenVariableDict] = None

  def _bind(
      self,
      intermediate_dict: flax.core.scope.FrozenVariableDict,
      module: module_lib.Module,
  ) -> None:
    """Bind the intermediate context to self."""
    if self._intermediate_dict is not None:
      # Nested `context.set_in_call()`. Should not be possible.
      raise RuntimeError('Intermediate context already set.')
    if self._module is not module:
      raise ValueError(
          'Intermediate capture and call instances do not match:'
          f' {self._module.name} ({self._module(module).__name__}) vs'
          f' {module.name} ({type(module).__name__})'
      )
    self._intermediate_dict = intermediate_dict

  def _finalize(self) -> None:
    """Create the nested intermediate proxies."""
    assert not self._finalized
    self._finalized = True
    cache: dict[int, _ModuleProxy] = {}
    # Traverse tree twice:
    # 1. To create the proxies
    # 2. To set the intermediate values (can't be done in `1` as futures not
    # yet created)
    self._proxy = traverse.recursive_replace(  # pytype: disable=annotation-type-mismatch
        self._module,
        replace_fn=functools.partial(_ModuleProxy.from_module, cache=cache),
    )
    intermediate_dict = self._intermediate_dict
    assert intermediate_dict is not None
    intermediate_dict = flax.core.unfreeze(intermediate_dict)
    traverse.recursive_replace(
        self._module,
        replace_fn=functools.partial(
            _set_intermediate_values,
            cache=cache,
            intermediate_dict=intermediate_dict,
        ),
    )
    assert not intermediate_dict  # Intermediate dict should have been cleared
    self._intermediate_dict = None
    self._module = None

  def __getattr__(self, name):
    if not self._finalized:
      raise AttributeError(
          'Cannot access intermediate values from within the'
          ' `capture_intermediates()` contextmanager.'
      )
    else:
      return getattr(self._proxy, name)

  def __getitem__(self, key: str) -> Any:
    if not self._finalized:
      raise KeyError(
          'Cannot access intermediate values from within the'
          ' `capture_intermediates()` contextmanager.'
      )
    else:
      return self._proxy.__getitem__(key)

  def __repr__(self) -> str:
    if not self._finalized:
      return f'{type(self).__name__}()'
    else:
      return self._proxy.__repr__()

  def __dir__(self) -> list[str]:
    """List attributes for Colab support."""
    if not self._finalized:
      return []
    else:
      return self._proxy.__dir__()

  def tree_flatten(
      self,
  ) -> tuple[list[Any], ModuleIntermediateProxy]:
    """`jax.tree_utils` support."""
    if not self._finalized:
      raise ValueError(
          'Cannot pass intermediates to tree_utils inside the'
          ' `capture_intermediates`'
      )

    # TODO(epot): This does not support model sharing (as tree_utils will
    # duplicate the node.)
    return ([self._proxy], self)  # pytype: disable=bad-return-type

  @classmethod
  def tree_unflatten(
      cls,
      metadata: ModuleIntermediateProxy,
      array_field_values: list[Any],
  ) -> ModuleIntermediateProxy:
    (proxy,) = array_field_values
    self = cls(metadata._module)  # pylint: disable=protected-access
    self._finalized = True
    self._proxy = proxy  # pylint: disable=protected-access
    return self


def _without_kwargs(fn: _FnT) -> _FnT:
  """Do not forward kwargs."""

  @functools.wraps(fn)
  def decorated(
      module: _T,
      *,
      attributes: dict[str, Any],
      name: Optional[str],
      future_parent: traverse.Future[module_lib.Module],
      **kwargs,
  ) -> _T:
    """."""
    del attributes, name, future_parent
    fn(module, **kwargs)

  return decorated


@_without_kwargs
def _set_intermediate_values(
    module: module_lib.Module,
    *,
    cache: _Cache,
    intermediate_dict: dict[str, Any],
) -> module_lib.Module:
  """."""
  proxy = cache[id(module)]
  proxy._set_intermediate(intermediate_dict)  # pylint: disable=protected-access
  return module


_Cache: TypeAlias = dict[int, _ModuleProxy]
