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

"""Fake import utils."""

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import functools
import inspect
import types
from typing import Callable, Iterator, Optional

from kauldron.konfig import configdict_proxy


@contextlib.contextmanager
def imports() -> Iterator[None]:
  """Contextmanager which replace import statements by configdicts.

  Usage:

  ```python
  with konfig.imports():
    import xyz.abc as a0

  a0.MyClass(x=123) == configdict.ConfigDict(
      x=123,
      __qualname__='xyz.abc.MyClass',
  )
  ```

  Yields:
    None
  """
  new_import = functools.partial(
      _fake_import,
      proxy_cls=configdict_proxy.ConfigDictProxyObject,
      origin_import=builtins.__import__,
  )
  with _fake_imports(new_import=new_import):
    yield


@contextlib.contextmanager
def mock_modules(*module_names: str):
  """Contextmanager which replaces list of modules with ConfigDictProxyObjects.

  Meant for updating configs in an interactive environments where
  `with konfig.imports()` would be inconvenient, because the modules in
  question should remain usable and not be globally replaced by
  ConfigDictProxyObjects.

  Example:
  ```
  from kauldron import kd

  cfg = ...  # import or construct a konfig.ConfigDict instance

  with kd.konfig.mock_modules():
    cfg.losses["l1"] = kd.losses.L1(preds="preds.image", targets="batch.image")
    # cfg.losses["l1"] is a konfig.ConfigDict rather than a kd.losses.L1

  l1 = kd.losses.L1(preds="preds.image", targets="batch.image")
  # l1 is still a kd.losses.L1 instance
  ```

  Args:
    *module_names: (optional) if given, only modules given here will be mocked
      to act like in `konfig.imports()` context. By default mock all modules.
      Imported alias should be givel (rather than the full name of the module).
      E.g. should be `np` instead of `numpy` if using `import numpy as np`.

  Yields:
    None
  """
  # Use the globals of the frame of the caller (two steps up because of the
  # contextlib decorator)
  global_ns = inspect.stack()[2].frame.f_globals

  if not module_names:
    modules = {  # By default, replace all modules
        name: module
        for name, module in global_ns.items()
        if isinstance(module, types.ModuleType)
    }
  else:
    modules = {name: global_ns[name] for name in module_names}

  # Do not replace `konfig` (note that `kd.konfig` will be mocked)
  # Filter `ecolab` for `xxx;` compatibility
  for name in ['konfig', 'ecolab']:
    modules.pop(name, None)

  # Create the new fake modules
  new_modules = {
      k: configdict_proxy.ConfigDictProxyObject.from_module_name(
          _get_module_name(m)
      )
      for k, m in modules.items()
  }
  try:
    global_ns.update(new_modules)
    yield
  finally:
    # Restore the original modules
    global_ns.update(modules)


def _get_module_name(module: types.ModuleType) -> str:
  """Returns the name of the module."""
  try:
    name = module.__dict__['__name__']  # Do not trigger lazy-imports
  except KeyError:  # Likely a lazy_imports
    if module.__module__ != 'etils.ecolab.lazy_utils':
      raise ValueError(f'Unexpected module: {module}') from None
    name = module._etils_state.module_name  # pylint: disable=protected-access
  return name


@contextlib.contextmanager
def _fake_imports(
    *,
    new_import: Callable[..., types.ModuleType],
) -> Iterator[None]:
  """Contextmanager which replace import statements by dummy `ProxyObject`.

  Usage:

  ```python
  with konfig.fake_imports(new_import=...):
    import xyz.abc as a0

  assert isinstance(a0, ProxyObject)
  ```

  Args:
    new_import: New import to replace

  Yields:
    None
  """
  # Need to mock `__import__` (instead of `sys.meta_path`, as we do not want
  # to modify the `sys.modules` cache in any way)
  origin_import = builtins.__import__
  try:
    builtins.__import__ = new_import
    yield
  finally:
    builtins.__import__ = origin_import


def _fake_import(
    name: str,
    globals_=None,
    locals_=None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
    *,
    proxy_cls: type[ProxyObject],
    origin_import: Callable[..., types.ModuleType],
) -> types.ModuleType:
  """Mock of `builtins.__import__`.

  Args:
    name: Module to import
    globals_: Same as `builtins.__import__`
    locals_: Same as `builtins.__import__`
    fromlist: Same as `builtins.__import__`
    level: Same as `builtins.__import__`
    proxy_cls: The module/class/function type of the fake import statement.
      Allow to control the behavior of the fake_imports.
    origin_import: Original import function

  Returns:
    The `ProxyObject` fake module
  """
  del globals_, locals_  # Unused

  if level:
    raise ValueError(f'Relative import statements not supported ({name}).')

  root_name, *parts = name.split('.')
  root = proxy_cls.from_cache(name=root_name)
  root.is_import = True
  if not fromlist:
    # import x.y.z
    # import x.y.z as z

    # Register the child modules
    child = root
    for name in parts:
      child = child.child_import(name)

    _maybe_import(origin_import, child.qualname)
    return root
  else:
    # from x.y.z import a, b

    # Return the inner-most module
    for name in parts:
      root = root.child_import(name)
    # Register the child imports
    for name in fromlist:
      child = root.child_import(name)
      _maybe_import(origin_import, child.qualname)
    return root
  return root


@dataclasses.dataclass(eq=False, kw_only=True)
class ProxyObject:
  """Base class to represent a module, function,..."""

  name: str
  parent: Optional[ProxyObject] = None
  # Whether or not the attribute was from an import or not:
  # `import a.b.c` vs `import a ; a.b.c`
  is_import: bool = False

  @classmethod
  @functools.cache
  def from_cache(cls, **kwargs):
    """Factory to cache all instances of module.

    Note: The cache is global to all instances of the
    `fake_import` contextmanager.

    Args:
      **kwargs: Init kwargs

    Returns:
      New object
    """
    return cls(**kwargs)

  @property
  def qualname(self) -> str:
    if not self.parent:
      return self.name

    if self.parent.is_import and not self.is_import:
      separator = ':'
    else:
      separator = '.'

    return f'{self.parent.qualname}{separator}{self.name}'

  def __repr__(self) -> str:
    return f'{type(self).__name__}({self.qualname!r})'

  def __getattr__(self, name: str) -> ProxyObject:
    return type(self).from_cache(
        name=name,
        parent=self,
    )

  def child_import(self, name: str) -> ProxyObject:
    """Returns the child import."""
    obj = getattr(self, name)
    # The cache is shared, so the 2 objects are the same:
    # import a ; a.b.c
    # from a.b import c
    obj.is_import = True
    return obj

  def __call__(self, *args, **kwargs):
    raise NotImplementedError('Inherit ProxyObject to support `__call__`.')


_LAZY_IMPORTED_MODULES: tuple[str, ...] | None = None
_IMPORTED_MODULES: tuple[str, ...] | None = None


@contextlib.contextmanager
def set_lazy_imported_modules(
    *,
    include: list[str],
    exclude: list[str] | tuple[str, ...] = (),
) -> Iterator[None]:
  """Set which modules inside `with konfig.imports()` will be lazy-imported."""
  global _LAZY_IMPORTED_MODULES
  global _IMPORTED_MODULES
  assert _LAZY_IMPORTED_MODULES is None
  assert _IMPORTED_MODULES is None
  _LAZY_IMPORTED_MODULES = tuple(include)
  _IMPORTED_MODULES = tuple(exclude)
  try:
    yield
  finally:
    _LAZY_IMPORTED_MODULES = None
    _IMPORTED_MODULES = None


def _maybe_import(import_fn, module_name: str) -> None:
  """Trigger the actual import."""
  with _fake_imports(new_import=import_fn):
    if _LAZY_IMPORTED_MODULES is None:
      import_fn(module_name)
      return
    assert _IMPORTED_MODULES is not None
    if _module_name_is_in(module_name, _IMPORTED_MODULES):
      import_fn(module_name)  # Module explicitly set as imported
    if _module_name_is_in(module_name, _LAZY_IMPORTED_MODULES):
      pass  # Module explicitly set as lazy
    else:
      # By default, import everything that was not explicitly set as lazy
      import_fn(module_name)


def _module_name_is_in(module_name: str, names: tuple[str, ...]) -> bool:
  if '*' in names:
    return True
  return module_name in names or f'{module_name}.'.startswith(
      tuple(f'{n}.' for n in names)
  )
