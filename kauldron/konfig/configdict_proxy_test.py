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

"""Tests for configdict_proxy."""

import pathlib
import types

from etils import epy
from kauldron import konfig
import pytest


@konfig.set_lazy_imported_modules(include=["*"])
def test_configdict():
  with konfig.imports():
    import abc.edf as some_module  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  assert some_module.MyClass(
      x=123,
      y=some_module.Other(
          None,
          123,
          a=[
              some_module.X(),
              some_module.X(),
          ],
      ),
  ) == konfig.ConfigDict({
      "__qualname__": "abc.edf:MyClass",
      "x": 123,
      "y": konfig.ConfigDict({
          "__qualname__": "abc.edf:Other",
          "0": None,
          "1": 123,
          "a": [
              konfig.ConfigDict({"__qualname__": "abc.edf:X"}),
              konfig.ConfigDict({"__qualname__": "abc.edf:X"}),
          ],
      }),
  })


def test_configdict_resolve_constructor():
  with konfig.imports():
    import types as fake_types  # pylint: disable=reimported,g-import-not-at-top  # pytype: disable=import-error
    import pathlib as fake_pathlib  # pylint: disable=reimported,g-import-not-at-top  # pytype: disable=import-error

  cfg = fake_types.SimpleNamespace(
      x=123,
      const=fake_pathlib.Path,
      y=fake_types.SimpleNamespace(
          elems=[
              fake_types.SimpleNamespace(),
              fake_types.SimpleNamespace(),
          ],
      ),
  )
  expected_cfg = konfig.ConfigDict({
      "__qualname__": "types:SimpleNamespace",
      "x": 123,
      "const": {"__const__": "pathlib:Path"},
      "y": konfig.ConfigDict({
          "__qualname__": "types:SimpleNamespace",
          "elems": [
              konfig.ConfigDict({"__qualname__": "types:SimpleNamespace"}),
              konfig.ConfigDict({"__qualname__": "types:SimpleNamespace"}),
          ],
      }),
  })

  assert cfg == expected_cfg
  assert repr(cfg) == epy.dedent("""
      <ConfigDict[types.SimpleNamespace(
          const=pathlib.Path,
          x=123,
          y=types.SimpleNamespace(
              elems=[
                  types.SimpleNamespace(),
                  types.SimpleNamespace(),
              ],
          ),
      )]>
      """)

  obj = konfig.resolve(expected_cfg)
  assert isinstance(obj, types.SimpleNamespace)
  assert obj.x == 123
  assert obj.const is pathlib.Path
  assert obj.y == types.SimpleNamespace(
      elems=(types.SimpleNamespace(), types.SimpleNamespace())
  )

  obj = konfig.resolve(fake_pathlib.Path("a", "b"))  # pytype: disable=wrong-arg-types
  expected_obj = pathlib.Path("a", "b")
  assert obj == expected_obj

  obj = konfig.resolve(
      fake_pathlib.Path(
          "a",
          fake_pathlib.Path("b"),
          "c",
      )
  )  # pytype: disable=wrong-arg-types
  expected_obj = pathlib.Path("a", pathlib.Path("b"), "c")
  assert obj == expected_obj


def test_configdict_args_mutation():
  with konfig.imports():
    import pathlib as fake_pathlib  # pylint: disable=reimported,g-import-not-at-top  # pytype: disable=import-error

  obj = fake_pathlib.Path("a", "b")  # pytype: disable=wrong-arg-types
  # pytype: disable=unsupported-operands
  assert obj[0] == "a"
  assert obj[1] == "b"
  assert obj[-1] == "b"
  obj[-1] = "b2"  # pylint: disable=unsupported-assignment-operation
  assert obj[-1] == "b2"

  with pytest.raises(IndexError):
    _ = obj[-3]

  with pytest.raises(IndexError):
    _ = obj[2]

  obj[2] = "c1"  # pylint: disable=unsupported-assignment-operation
  # pytype: enable=unsupported-operands

  expected_obj = pathlib.Path("a", "b2", "c1")
  assert konfig.resolve(obj) == expected_obj
