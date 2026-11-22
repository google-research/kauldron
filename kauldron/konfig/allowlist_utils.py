# Copyright 2026 The kauldron Authors.
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

"""Allowlist restricting which symbols `konfig.resolve` is allowed to import.

Resolving a config imports and calls the objects referenced by the
`__qualname__` / `__const__` entries of the config. This is fine when the
config was created by the program itself (a `konfig.imports()` config file is
regular Python code, so it is as trusted as the binary running it), but not
when the config was deserialized from an untrusted source (e.g. a
`config.json` read from a directory writable by a large group).

For those cases, `konfig.resolve` accepts an `allowlist=` of the modules and
symbols the config is allowed to use:

```python
trainer = konfig.resolve(cfg, allowlist=[kd, my_project.models])
```

The allowlist is enforced before anything is imported, so a rejected config
never executes any of its symbols.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
import types
from typing import Any, Union


# Entries accepted in `konfig.resolve(..., allowlist=[...])`: a dotted prefix
# (`'kauldron.data'`), a module (`kd.data`), a `konfig.imports()` symbol, or any
# object exposing `__module__`/`__qualname__` (class, function,...).
AllowlistEntry = Any
AllowlistArg = Union['Allowlist', Sequence[AllowlistEntry]]


class NotAllowedError(ValueError):
  """Raised when a config resolves a symbol missing from the allowlist."""


def normalize_qualname(qualname: str) -> str:
  """Normalizes `a.b:C.D` into the plain dotted path `a.b.C.D`."""
  return qualname.replace(':', '.')


@dataclasses.dataclass(frozen=True)
class Allowlist:
  """Set of dotted prefixes a config is allowed to import.

  A qualname is allowed if it is equal to, or nested inside, one of the
  prefixes. Matching is done on dotted-path components, so the prefix
  `kauldron` allows `kauldron.data:Pipeline` but not `kauldron_evil:Exploit`.

  Attributes:
    prefixes: The normalized dotted prefixes (see `normalize_qualname`).
  """

  prefixes: tuple[str, ...]

  @classmethod
  def from_arg(cls, allowlist: AllowlistArg) -> Allowlist:
    """Normalizes the user-provided `allowlist=` argument."""
    if isinstance(allowlist, Allowlist):
      return allowlist
    if isinstance(allowlist, str):
      raise TypeError(
          '`allowlist=` should be a sequence of modules/symbols/str, not a'
          f' single `str`. Got {allowlist!r}. Did you mean'
          f' `allowlist=[{allowlist!r}]` ?'
      )
    return cls(prefixes=tuple(sorted({_entry_to_prefix(e) for e in allowlist})))

  def is_allowed(self, qualname: str) -> bool:
    """Returns `True` if the given config qualname can be imported."""
    qualname = normalize_qualname(qualname)
    if _has_dunder_part(qualname):
      return False
    return any(
        qualname == prefix or qualname.startswith(f'{prefix}.')
        for prefix in self.prefixes
    )

  def assert_allowed(self, qualnames: Iterable[str]) -> None:
    """Raises `NotAllowedError` if any of the qualnames is not allowed."""
    qualnames = set(qualnames)

    # Dunder attributes allow escaping the allowlisted module (e.g. through
    # `__globals__`, `__class__`,...), so are always rejected.
    dunders = sorted(
        q for q in qualnames if _has_dunder_part(normalize_qualname(q))
    )
    if dunders:
      raise NotAllowedError(
          'Config cannot resolve dunder attributes (they allow escaping the'
          f' allowlist): {dunders}'
      )

    rejected = sorted(q for q in qualnames if not self.is_allowed(q))
    if rejected:
      raise NotAllowedError(
          f'Config is not allowed to import: {rejected}\n'
          f'Allowed prefixes: {list(self.prefixes)}\n'
          'Resolving a config imports and calls arbitrary Python symbols, so'
          ' an `allowlist=` was set to restrict which symbols this config can'
          ' use. If those symbols are trusted, add them to the'
          ' `konfig.resolve(..., allowlist=[...])` argument.'
      )


def _entry_to_prefix(entry: AllowlistEntry) -> str:
  """Converts a single user-provided allowlist entry to a dotted prefix."""
  # `konfig.imports()` symbols are duck-typed (rather than imported from
  # `fake_import_utils`) to avoid a circular dependency.
  proxy_qualname = getattr(entry, 'qualname', None)

  if isinstance(entry, str):
    prefix = entry
  elif isinstance(entry, types.ModuleType):
    prefix = entry.__name__
  elif isinstance(proxy_qualname, str):  # `with konfig.imports():` symbol
    prefix = proxy_qualname
  else:  # Class, function,...
    module = getattr(entry, '__module__', None)
    qualname = getattr(entry, '__qualname__', None)
    if not isinstance(module, str) or not isinstance(qualname, str):
      raise TypeError(
          f'Unsupported `allowlist=` entry: {entry!r}. Expected a module, a'
          ' class/function, a `konfig.imports()` symbol or a `str` prefix.'
      )
    prefix = f'{module}.{qualname}'

  prefix = normalize_qualname(prefix)
  if not all(part.isidentifier() for part in prefix.split('.')):
    raise ValueError(
        f'Invalid `allowlist=` entry: {entry!r} (normalized to {prefix!r}).'
        ' Only top-level modules, classes and functions are supported.'
    )
  return prefix


def _has_dunder_part(qualname: str) -> bool:
  return any(part.startswith('__') for part in qualname.split('.'))
