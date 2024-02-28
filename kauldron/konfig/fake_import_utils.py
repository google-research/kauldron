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

"""Fake import utils."""

from __future__ import annotations

import builtins
import contextlib
import ctypes
import dataclasses
import functools
import inspect
import types
from typing import Any, Callable, Iterator, NoReturn, Optional

from etils import edc
from etils import epy
from kauldron.konfig import configdict_proxy


@contextlib.contextmanager
def imports(*, lazy: bool = False) -> Iterator[None]:
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

  Args:
    lazy: If `True`, import won't be directly executed, but only resolved when
      calling `konfig.resolve`. This allow to import the config even when some
      dependencies are missing.

  Yields:
    None
  """
  new_import = functools.partial(
      _fake_import,
      proxy_cls=configdict_proxy.ConfigDictProxyObject,
      origin_import=builtins.__import__,
  )
  if lazy:
    lazy_cm = set_lazy_imported_modules()
  else:
    lazy_cm = contextlib.nullcontext()

  with lazy_cm, _fake_imports(new_import=new_import):
    try:
      yield
    except ImportError as e:
      epy.reraise(
          e,
          suffix=(
              '\nIf those imports are not resolved here (e.g. XManager info'
              ' inside the Kauldron trainer config), those modules should be'
              ' imported in a separate `with konfig.imports(lazy=True)` scope.'
          ),
      )


@contextlib.contextmanager
def mock_modules():
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

  Yields:
    None
  """
  # Use the globals of the frame of the caller (two steps up because of the
  # contextlib decorator)
  frame = inspect.stack()[2].frame
  with (
      _mock_modules(frame=frame, property_name='f_globals'),
      _mock_modules(frame=frame, property_name='f_locals'),
  ):
    yield


@contextlib.contextmanager
def _mock_modules(frame: types.FrameType, property_name: str):
  """Mock module implementation."""
  global_ns: dict[str, Any] = getattr(frame, property_name)

  modules = {  # By default, replace all modules
      name: module
      for name, module in global_ns.items()
      if isinstance(module, types.ModuleType)
  }
  # Do not replace `konfig` (note that `kd.konfig` will be mocked)
  # Filter `ecolab` for `xxx;` compatibility
  for name in ['konfig', 'ecolab']:
    modules.pop(name, None)

  # Create the new fake modules
  new_modules = {
      k: configdict_proxy.ConfigDictProxyObject.from_module_name(module_name)
      for k, m in modules.items()
      if (module_name := _get_module_name(m))
  }
  try:
    global_ns.update(new_modules)
    _force_locals_update(frame, property_name)
    yield
  finally:
    # Restore the original modules
    global_ns.update(modules)
    _force_locals_update(frame, property_name)


def _force_locals_update(frame, property_name):
  if property_name != 'f_locals':
    return
  # In the current Python version, mutating the `locals()` has not effect.
  # This is a hack to force update:
  # https://stackoverflow.com/questions/34650744/modify-existing-variable-in-locals-or-frame-f-locals
  # Future Python version likely won't require this:
  # https://peps.python.org/pep-0667/
  ctypes.pythonapi.PyFrame_LocalsToFast(
      ctypes.py_object(frame),
      ctypes.c_int(0),  # Keep existing `locals()`
  )


def _get_module_name(module: types.ModuleType) -> str | None:
  """Returns the name of the module."""
  getattr_ = inspect.getattr_static
  if module_name := getattr_(module, '__name__', None):
    return module_name
  # Otherwise the module don't have a `__name__`
  if getattr_(module, '__module__', '') == 'etils.ecolab.lazy_utils':
    return module._etils_state.module_name  # lazy_imports  # pylint: disable=protected-access
  if getattr_(module, '__etils_invalidated__', False):
    return None  # Invalidated module, do not return

  raise ValueError(f'Cannot mock unexpected module: {module}')


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

  @property
  def __mro_entries__(self) -> NoReturn:
    raise ValueError(
        "You're trying to inherit from a `konfig` object, rather than an"
        f' actual Python class: {self.qualname!r}. Likely because the module '
        'was imported inside a `konfig.imports()` context.'
    )


@dataclasses.dataclass(frozen=True)
class _LazyImportState:
  lazy_imported_modules: tuple[str, ...]
  except_modules: tuple[str, ...]


@edc.dataclass
@dataclasses.dataclass(frozen=True)
class _LazyImportStack:
  stack: edc.ContextVar[list[_LazyImportState]] = dataclasses.field(
      default_factory=list
  )


_lazy_import_stack = _LazyImportStack()


@contextlib.contextmanager
def set_lazy_imported_modules(
    lazy_import: list[str] | tuple[str, ...] = ('*',),
    *,
    except_: list[str] | tuple[str, ...] = (),
) -> Iterator[None]:
  """Set which modules inside `with konfig.imports()` will be lazy-imported."""
  assert isinstance(lazy_import, (list, tuple))
  assert isinstance(except_, (list, tuple))

  _lazy_import_stack.stack.append(
      _LazyImportState(
          lazy_imported_modules=tuple(lazy_import),
          except_modules=tuple(except_),
      )
  )
  try:
    yield
  finally:
    _lazy_import_stack.stack.pop()


def _maybe_import(origin_import, module_name: str) -> None:
  """Trigger the actual import."""
  # Restore the original import
  with _fake_imports(new_import=origin_import):
    # Lazy import not set, so import everything
    if not _lazy_import_stack.stack:
      origin_import(module_name)
      return

    info = _lazy_import_stack.stack[-1]
    # TODO(epot): Here, except could overwrite a lazy-import set before in the
    # stack. Instead, the stack should only contain the most restrictive set
    # of exclude.
    if _module_name_is_in(module_name, info.except_modules):
      # Except takes precedence over lazy imports
      origin_import(module_name)
    elif _module_name_is_in(module_name, info.lazy_imported_modules):
      pass  # Module explicitly set as lazy
    else:
      # By default, import everything that was not explicitly set as lazy
      origin_import(module_name)


def _module_name_is_in(module_name: str, names: tuple[str, ...]) -> bool:
  if '*' in names:
    return True
  return module_name in names or f'{module_name}.'.startswith(
      tuple(f'{n}.' for n in names)
  )
