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

"""Fake import test."""

from __future__ import annotations

from kauldron import konfig


def assert_module(
    m: konfig.fake_import_utils.ProxyObject, name: str
) -> None:
  assert isinstance(m, konfig.fake_import_utils.ProxyObject)
  assert m.qualname == name


def test_fake_imports():
  with konfig.imports():
    # pylint: disable=g-import-not-at-top,g-multiple-import
    # pytype: disable=import-error
    import a0
    import a1.b.c
    import a2.b.c as c00
    import a2.b.c as c01  # Importing object twice is the same instance
    from a2.b import c as c02

    from a3 import c2, c3
    from a3.b.c import c4
    # pytype: enable=import-error
    # pylint: enable=g-import-not-at-top,g-multiple-import

  assert_module(a0, 'a0')
  assert_module(a1.b.c, 'a1.b.c')
  assert_module(a1.non_module.c, 'a1:non_module.c')
  assert_module(c00, 'a2.b.c')
  assert_module(c01, 'a2.b.c')
  assert_module(c02, 'a2.b.c')
  assert_module(c02.non_module.c, 'a2.b.c:non_module.c')
  assert_module(c2, 'a3.c2')
  assert_module(c3, 'a3.c3')
  assert_module(c3.non_module.c, 'a3.c3:non_module.c')
  assert_module(c4, 'a3.b.c.c4')
  assert_module(c4.non_module.c, 'a3.b.c.c4:non_module.c')
  assert c01 is c00
  assert c02 is c00
