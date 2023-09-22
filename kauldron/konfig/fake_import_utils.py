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
from typing import Iterator, Optional
import unittest.mock

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
  with _fake_imports(proxy_cls=configdict_proxy.ConfigDictProxyObject):
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

  with kd.konfig.mock_modules("kd", "nn"):
    cfg.losses["l1"] = kd.losses.L1(preds="preds.image", targets="batch.image")
    # cfg.losses["l1"] is a konfig.ConfigDict rather than a kd.losses.L1

  l1 = kd.losses.L1(preds="preds.image", targets="batch.image")
  # l1 is still a kd.losses.L1 instance
  ```

  Args:
    *module_names: names of already imported modules to be replaced with
      ConfigDictProxyObjects inside the context. Should be the name of the
      module as used in the globals() (rather than the full name of the module).
      E.g. should be `np` instead of `numpy` if using `import numpy as np`.

  Yields:
    None
  """
  try:
    # If using Notebook/Colab get the globals from there
    import IPython  # pylint: disable=g-import-not-at-top
    global_ns = IPython.get_ipython().kernel.shell.user_ns
  except (ImportError, NameError, AttributeError):
    # otherwise use the globals of the frame of the caller
    # (two steps up because of the contextlib decorator)
    global_ns = inspect.stack()[2].frame.f_globals

  with contextlib.ExitStack() as stack:
    for name in module_names:
      local_name = f'__main__.{name}'
      module_name = global_ns.get(name).__name__

      # Extract the sub-module
      root_name, *parts = module_name.split('.')
      root = configdict_proxy.ConfigDictProxyObject.from_cache(name=root_name)
      root.is_import = True
      for name in parts:
        root = root.child_import(name)

      stack.enter_context(unittest.mock.patch(local_name, root))
    yield


@contextlib.contextmanager
def _fake_imports(
    *,
    proxy_cls: Optional[type[ProxyObject]] = None,
) -> Iterator[None]:
  """Contextmanager which replace import statements by dummy `ProxyObject`.

  Usage:

  ```python
  with konfig.fake_imports(proxy_cls=ProxyObject):
    import xyz.abc as a0

  assert isinstance(a0, ProxyObject)
  ```

  Args:
    proxy_cls: The module/class/function type of the fake import statement.
      Allow to control the behavior of the fake_imports.

  Yields:
    None
  """
  proxy_cls = proxy_cls or ProxyObject

  # Need to mock `__import__` (instead of `sys.meta_path`, as we do not want
  # to modify the `sys.modules` cache in any way)
  origin_import = builtins.__import__
  try:
    builtins.__import__ = functools.partial(_fake_import, proxy_cls=proxy_cls)
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
):
  """Mock of `builtins.__import__`."""
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
    childs = root
    for name in parts:
      childs = childs.child_import(name)

    return root
  else:
    # from x.y.z import a, b

    # Return the inner-most module
    for name in parts:
      root = root.child_import(name)
    # Register the child imports
    for name in fromlist:
      root.child_import(name)
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
