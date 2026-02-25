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

from kauldron import konfig
from kauldron.konfig import py_flag_utils


def test_simple_call():
  result = py_flag_utils.parse_py_flag_value('types.SimpleNamespace(x=1)')
  assert result == konfig.ConfigDict({
      '__qualname__': 'types:SimpleNamespace',
      'x': 1,
  })


def test_call_with_positional_args():
  result = py_flag_utils.parse_py_flag_value('pathlib.Path("a", "b")')
  assert result == konfig.ConfigDict({
      '__qualname__': 'pathlib:Path',
      '0': 'a',
      '1': 'b',
  })


def test_call_no_args():
  result = py_flag_utils.parse_py_flag_value('types.SimpleNamespace()')
  assert result == konfig.ConfigDict({
      '__qualname__': 'types:SimpleNamespace',
  })


def test_const_ref():
  result = py_flag_utils.parse_py_flag_value('pathlib.Path')
  assert result == konfig.ConfigDict({
      '__const__': 'pathlib:Path',
  })


def test_deep_module_path():
  result = py_flag_utils.parse_py_flag_value(
      'path.to.project.module.MyClass(x=1)'
  )
  assert result == konfig.ConfigDict({
      '__qualname__': 'path.to.project.module:MyClass',
      'x': 1,
  })


def test_deep_const_ref():
  result = py_flag_utils.parse_py_flag_value(
      'path.to.project.module.MyEnum.VALUE'
  )
  assert result == konfig.ConfigDict({
      '__const__': 'path.to.project.module.MyEnum:VALUE',
  })


def test_explicit_colon_qualname():
  result = py_flag_utils.parse_py_flag_value('path.to.module:MyEnum.VALUE')
  assert result == konfig.ConfigDict({
      '__const__': 'path.to.module:MyEnum.VALUE',
  })


def test_explicit_colon_call():
  result = py_flag_utils.parse_py_flag_value('path.to.module:MyClass(x=1)')
  assert result == konfig.ConfigDict({
      '__qualname__': 'path.to.module:MyClass',
      'x': 1,
  })


def test_list():
  result = py_flag_utils.parse_py_flag_value(
      '[types.SimpleNamespace(), 4, "hello"]'
  )
  assert result == [
      konfig.ConfigDict({'__qualname__': 'types:SimpleNamespace'}),
      4,
      'hello',
  ]


def test_tuple():
  result = py_flag_utils.parse_py_flag_value('(1, 2, 3)')
  assert result == (1, 2, 3)


def test_dict():
  result = py_flag_utils.parse_py_flag_value(
      '{"key": types.SimpleNamespace(x=1)}'
  )
  assert result == {
      'key': konfig.ConfigDict({
          '__qualname__': 'types:SimpleNamespace',
          'x': 1,
      }),
  }


def test_nested_call():
  result = py_flag_utils.parse_py_flag_value(
      'types.SimpleNamespace(inner=types.SimpleNamespace(x=1))'
  )
  assert result == konfig.ConfigDict({
      '__qualname__': 'types:SimpleNamespace',
      'inner': konfig.ConfigDict({
          '__qualname__': 'types:SimpleNamespace',
          'x': 1,
      }),
  })


def test_maybe_parse_non_py_prefix():
  assert py_flag_utils.maybe_parse_py_flag_value('hello') == 'hello'
  assert py_flag_utils.maybe_parse_py_flag_value(42) == 42
  assert py_flag_utils.maybe_parse_py_flag_value(None) is None


def test_maybe_parse_py_prefix():
  result = py_flag_utils.maybe_parse_py_flag_value(
      'py::types.SimpleNamespace(x=1)'
  )
  assert result == konfig.ConfigDict({
      '__qualname__': 'types:SimpleNamespace',
      'x': 1,
  })


def test_maybe_parse_py_prefix_alias():
  result = py_flag_utils.maybe_parse_py_flag_value('py::kd.data.Xxx(x=1)')
  assert result == konfig.ConfigDict({
      '__qualname__': 'kauldron.kd:data.Xxx',
      'x': 1,
  })


def test_literals():
  assert py_flag_utils.parse_py_flag_value('42') == 42
  assert py_flag_utils.parse_py_flag_value('"hello"') == 'hello'
  assert py_flag_utils.parse_py_flag_value('True') is True  # pylint: disable=g-bool-id-comparison
  assert py_flag_utils.parse_py_flag_value('None') is None
  assert py_flag_utils.parse_py_flag_value('3.14') == 3.14


def test_negative_number():
  assert py_flag_utils.parse_py_flag_value('-1') == -1
  assert py_flag_utils.parse_py_flag_value('-3.14') == -3.14


def test_mixed_args():
  result = py_flag_utils.parse_py_flag_value(
      'my.module.MyClass("pos_arg", x=1, y=True)'
  )
  assert result == konfig.ConfigDict({
      '__qualname__': 'my.module:MyClass',
      '0': 'pos_arg',
      'x': 1,
      'y': True,
  })
