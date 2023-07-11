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
from typing import Optional, Iterator

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
