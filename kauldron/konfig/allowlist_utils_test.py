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

import dataclasses
import types

from kauldron import konfig
from kauldron.konfig import allowlist_utils
import pytest


@dataclasses.dataclass
class MyObj:
  x: int = 0


def test_normalize_qualname():
  assert allowlist_utils.normalize_qualname('a.b:C.D') == 'a.b.C.D'
  assert allowlist_utils.normalize_qualname('a.b.C') == 'a.b.C'


def test_allowlist_from_str():
  allowlist = konfig.Allowlist.from_arg(['a.b', 'c'])
  assert allowlist.prefixes == ('a.b', 'c')

  assert allowlist.is_allowed('a.b')
  assert allowlist.is_allowed('a.b:C')
  assert allowlist.is_allowed('a.b.c:D.E')
  assert allowlist.is_allowed('c:D')

  assert not allowlist.is_allowed('a')
  assert not allowlist.is_allowed('a.bc:D')  # Not a path component prefix
  assert not allowlist.is_allowed('b.a:C')
  assert not allowlist.is_allowed('builtins:eval')


def test_allowlist_from_module():
  allowlist = konfig.Allowlist.from_arg([types])
  assert allowlist.prefixes == ('types',)
  assert allowlist.is_allowed('types:SimpleNamespace')
  assert not allowlist.is_allowed('typesx:SimpleNamespace')


def test_allowlist_from_symbol():
  allowlist = konfig.Allowlist.from_arg([MyObj])
  assert allowlist.prefixes == (f'{MyObj.__module__}.MyObj',)
  assert allowlist.is_allowed(f'{MyObj.__module__}:MyObj')
  assert not allowlist.is_allowed(f'{MyObj.__module__}:OtherObj')


def test_allowlist_from_proxy():
  with konfig.imports():
    import pathlib as fake_pathlib  # pylint: disable=reimported,g-import-not-at-top

  allowlist = konfig.Allowlist.from_arg([fake_pathlib])
  assert allowlist.prefixes == ('pathlib',)
  assert allowlist.is_allowed('pathlib:Path')
  assert not allowlist.is_allowed('pathlib2:Path')

  allowlist = konfig.Allowlist.from_arg([fake_pathlib.Path])
  assert allowlist.prefixes == ('pathlib.Path',)
  assert allowlist.is_allowed('pathlib:Path')
  assert not allowlist.is_allowed('pathlib:PurePath')


def test_allowlist_idempotent():
  allowlist = konfig.Allowlist.from_arg(['a'])
  assert konfig.Allowlist.from_arg(allowlist) is allowlist


def test_allowlist_empty():
  allowlist = konfig.Allowlist.from_arg([])
  assert not allowlist.is_allowed('a')
  assert not allowlist.is_allowed('')


def test_allowlist_reject_dunder():
  # Dunder are rejected, even nested inside an allowlisted module.
  allowlist = konfig.Allowlist.from_arg(['a'])
  assert not allowlist.is_allowed('a:b.__globals__.x')
  assert not allowlist.is_allowed('a.__class__:x')

  with pytest.raises(konfig.NotAllowedError, match='dunder'):
    allowlist.assert_allowed(['a:b.__globals__'])


def test_allowlist_single_str_error():
  with pytest.raises(TypeError, match='should be a sequence'):
    konfig.Allowlist.from_arg('a.b')


def test_allowlist_invalid_entry():
  with pytest.raises(TypeError, match='Unsupported'):
    konfig.Allowlist.from_arg([123])

  with pytest.raises(ValueError, match='Invalid'):
    konfig.Allowlist.from_arg(['a b'])


def test_assert_allowed():
  allowlist = konfig.Allowlist.from_arg(['a'])
  allowlist.assert_allowed([])
  allowlist.assert_allowed(['a:b', 'a.c:d'])

  with pytest.raises(konfig.NotAllowedError, match='not allowed to import'):
    allowlist.assert_allowed(['a:b', 'builtins:eval'])
