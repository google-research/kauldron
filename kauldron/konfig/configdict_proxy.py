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

"""Implementation of `ProxyObject` which resolve to `ConfigDict`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import functools
import importlib
import itertools
import typing
from typing import Any, TypeVar

from absl import logging
from etils import epy
from kauldron.konfig import configdict_base
from kauldron.konfig import fake_import_utils
from kauldron.konfig import utils
from kauldron.utils import immutabledict
import ml_collections


_T = TypeVar('_T')
_FnT = TypeVar('_FnT')

# * `{'__qualname__': 'xxx'}`: Resolved as `xxx()`
# * `{'__const__': 'xxx'}`: Resolved as `xxx`
QUALNAME_KEY = '__qualname__'
CONST_KEY = '__const__'


# `ConfigDictProxyObject` is a `dict`, so fake modules can be assinged as
# constant:
#
# with konfig.imports():
#   import my_module
#
# # `my_module == {'__const__': 'my_module'}`
# cfg = konfig.ConfigDict({'module': my_module})
# cfg = konfig.resolve(cfg)  # Resolve `my_module`
@dataclasses.dataclass(eq=False, repr=False)
class ConfigDictProxyObject(fake_import_utils.ProxyObject, dict):
  """Implementation of `ProxyObject` which resolve to `ConfigDict`."""

  def __new__(cls, items=None, **kwargs):
    # Support tree map functions
    # Inside `tree.map`, the classes are created as: `type(obj)(obj.items())`
    if items is not None:
      assert not kwargs
      return dict(items)
    return super().__new__(cls, **kwargs)

  def __post_init__(self):
    # `ConfigDictProxyObject` act as a constant, when assigned in another
    # configdict attribute
    dict.__init__(self, {CONST_KEY: self.qualname})

  @classmethod
  def from_module_name(cls, module_name: str) -> ConfigDictProxyObject:
    """Returns the proxy for the given module name.

    Args:
      module_name: Module name to import

    Returns:
      Proxy object
    """
    # Extract the sub-module
    root_name, *parts = module_name.split('.')
    root = cls.from_cache(name=root_name)
    root.is_import = True
    for name in parts:
      root = root.child_import(name)
    return root

  def __call__(self, *args, **kwargs) -> ml_collections.ConfigDict:
    """`my_module.MyObject()`."""
    args_kwargs = {str(i): v for i, v in enumerate(args)}
    return configdict_base.ConfigDict({
        QUALNAME_KEY: self.qualname,
        **args_kwargs,
        **kwargs,
    })

  # Overwritte `dict` methods
  def __bool__(self) -> bool:
    return True

  def __repr__(self) -> str:
    return f'ConfigDictProxyObject({self.qualname})'

  __eq__ = object.__eq__
  __hash__ = object.__hash__


@typing.overload
def resolve(cfg: ml_collections.ConfigDict, *, freeze: bool = ...) -> Any:
  ...


@typing.overload
def resolve(cfg: _T, *, freeze: bool = ...) -> _T:
  ...


def resolve(cfg, *, freeze=True):
  """Recursively parses a nested ConfigDict and resolves module constructors.

  Args:
    cfg: The config to resolved
    freeze: If `True` (default), `list` are converted to `tuple`,
      `dict`/`ConfigDict` are converted to `immutabledict`.

  Returns:
    The resolved config.
  """
  try:
    return _ConstructorResolver(freeze=freeze)._resolve_value(cfg)  # pylint: disable=protected-access
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.info(f'Full config (failing): {cfg}')  # pylint: disable=logging-fstring-interpolation
    epy.reraise(e, 'Error resolving the config:\n')


class _ConfigDictVisitor:
  """Class which recursivelly inspect/transform a ConfigDict.

  By default, the visitor is a no-op:

  ```python
  assert _ConfigDictVisitor.apply(cfg) == cfg
  ```

  Child can overwritte specific `_resolve_xyz` function to apply specific
  transformations.
  """

  def __init__(self, freeze=True):
    self._freeze = freeze
    self._types_to_resolver = {
        (dict, ml_collections.ConfigDict): self._resolve_dict,
        (list, tuple): self._resolve_sequence,
        ml_collections.FieldReference: self._resolve_reference,
    }

  def _resolve_value(self, value):
    """Apply the visitor/transformation to the config dict."""
    for cls, resolver_fn in self._types_to_resolver.items():
      if isinstance(value, cls):
        return resolver_fn(value)
    return self._resolve_leaf(value)  # Leaf value

  def _resolve_sequence(self, value):
    cls = type(value)
    if self._freeze:
      if cls not in (list, tuple):
        raise TypeError(f'Cannot freeze unknown sequence type {type(cls)}')
      cls = tuple
    return cls([
        _reraise_with_info(self._resolve_value, i)(v)
        for i, v in enumerate(value)
    ])

  def _resolve_dict(self, value):
    cls = type(value)
    if self._freeze:
      cls = immutabledict.ImmutableDict
    return cls({
        k: _reraise_with_info(self._resolve_value, k)(v)
        for k, v in _as_dict(value).items()
    })

  def _resolve_reference(self, value: ml_collections.FieldReference):
    return self._resolve_value(value.get())

  def _resolve_leaf(self, value):
    return value


class _ConstructorResolver(_ConfigDictVisitor):
  """Instanciate all `ConfigDict` proxy object."""

  def __init__(self, freeze=True):
    super().__init__(freeze=freeze)
    # Keep track of the constructed object, to allow the same object to be
    # defined twice.
    self._id_to_obj: dict[int, utils.CachedObj[Any]] = {}

  def _resolve_dict(self, value):
    # If the value was already resolved, return it (shared object)
    if id(value) in self._id_to_obj:
      return self._id_to_obj[id(value)].value

    # Dict proxies have `__const__` or `__qualname__` keys
    if QUALNAME_KEY in value:
      if CONST_KEY in value:
        raise ValueError(
            f'Conflict: Both {QUALNAME_KEY} and {CONST_KEY} are set. For'
            f' {value}'
        )
      qualname_key = QUALNAME_KEY
    elif CONST_KEY in value:
      qualname_key = CONST_KEY
    else:
      return super()._resolve_dict(value)

    kwargs = _as_dict(value)

    constructor = import_qualname(kwargs.pop(qualname_key))
    if hasattr(constructor, '__konfig_resolve_exclude_fields__'):
      exclude_fields = constructor.__konfig_resolve_exclude_fields__
    else:
      exclude_fields = ()

    if qualname_key == CONST_KEY:
      if kwargs:
        raise ValueError(
            f'Malformated constant: {kwargs}. Should only contain a single key.'
        )
      return constructor  # Constant are returned as-is

    kwargs = {
        k: (
            v
            if k in exclude_fields
            else _reraise_with_info(self._resolve_value, k)(v)
        )
        for k, v in kwargs.items()
    }
    args = [kwargs.pop(str(i)) for i in range(num_args(kwargs))]
    with epy.maybe_reraise(prefix=lambda: _make_cfg_error_msg(value)):
      obj = constructor(*args, **kwargs)
    # Allow the object to save the config it is comming from.
    if hasattr(type(obj), '__post_konfig_resolve__'):
      obj.__post_konfig_resolve__(value)
    self._id_to_obj[id(value)] = utils.CachedObj(ref=value, value=obj)
    return obj


def import_qualname(qualname_str: str) -> Callable[..., Any]:
  """Fix the import constructors."""
  match qualname_str.split(':'):
    case [import_str, attributes]:
      pass
    case [qualname_str]:  # Otherwise, assume single attribute
      import_str, attributes = qualname_str.rsplit('.', maxsplit=1)
    case _:
      raise ValueError(f'Invalid {qualname_str!r}')

  obj = importlib.import_module(import_str)
  for attr in attributes.split('.'):
    obj = getattr(obj, attr)
  return obj  # pytype: disable=bad-return-type


def num_args(obj: Mapping[str, Any]) -> int:
  """Returns the number of positional arguments of the callable."""
  for arg_id in itertools.count():
    if str(arg_id) not in obj:
      break
  return arg_id  # pylint: disable=undefined-loop-variable,undefined-variable


def _make_cfg_error_msg(cfg: ml_collections.ConfigDict) -> str:
  cfg_str = repr(cfg)
  cfg_str = cfg_str.removeprefix('<ConfigDict[').removesuffix(']>')
  if len(cfg_str) > 300:  # `textwrap.shorten` remove `\n` so don't use it
    cfg_str = cfg_str[:295] + '[...]'
  return f'Error while constructing cfg: {cfg_str}\n'


def _as_dict(values: Mapping[str, Any]) -> dict[str, Any]:
  """Convert to dict, reraising error message (for `FieldReference` errors)."""
  return {
      k: _reraise_with_info(lambda k: values[k], k)(k) for k in values.keys()
  }


def _reraise_with_info(fn: _FnT, info: str | int) -> _FnT:
  @functools.wraps(fn)
  def decorated(*args, **kwargs):
    try:
      return fn(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      info_ = f'[{info}]' if isinstance(info, int) else repr(info)
      epy.reraise(e, prefix=f'In {info_}:\n')

  return decorated
