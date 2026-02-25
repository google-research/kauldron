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

from __future__ import annotations

from kauldron import konfig
from kauldron.konfig import py_flag_parser
import pytest


# ---- Calls ----

_CALL_CASES = [
    pytest.param(
        'types.SimpleNamespace(x=1)',
        konfig.ConfigDict({
            '__qualname__': 'types:SimpleNamespace',
            'x': 1,
        }),
        id='simple',
    ),
    pytest.param(
        'types.SimpleNamespace()',
        konfig.ConfigDict({
            '__qualname__': 'types:SimpleNamespace',
        }),
        id='no_args',
    ),
    pytest.param(
        'pathlib.Path("a", "b")',
        konfig.ConfigDict({
            '__qualname__': 'pathlib:Path',
            '0': 'a',
            '1': 'b',
        }),
        id='positional_args',
    ),
    pytest.param(
        'my.module.MyClass("pos_arg", x=1, y=True)',
        konfig.ConfigDict({
            '__qualname__': 'my.module:MyClass',
            '0': 'pos_arg',
            'x': 1,
            'y': True,
        }),
        id='mixed_args',
    ),
    pytest.param(
        'types.SimpleNamespace(inner=types.SimpleNamespace(x=1))',
        konfig.ConfigDict({
            '__qualname__': 'types:SimpleNamespace',
            'inner': konfig.ConfigDict({
                '__qualname__': 'types:SimpleNamespace',
                'x': 1,
            }),
        }),
        id='nested',
    ),
    pytest.param(
        'path.to.project.module.MyClass(x=1)',
        konfig.ConfigDict({
            '__qualname__': 'path.to.project.module:MyClass',
            'x': 1,
        }),
        id='deep_module_path',
    ),
    pytest.param(
        'my.mod.Cls(x=1,)',
        konfig.ConfigDict({
            '__qualname__': 'my.mod:Cls',
            'x': 1,
        }),
        id='trailing_comma',
    ),
    pytest.param(
        'kd.data.Xxx(x=1)',
        konfig.ConfigDict({
            '__qualname__': 'kauldron.kd:data.Xxx',
            'x': 1,
        }),
        id='alias_kd',
    ),
    pytest.param(
        'nn.Module(x=1)',
        konfig.ConfigDict({
            '__qualname__': 'flax.linen:Module',
            'x': 1,
        }),
        id='alias_nn',
    ),
]


@pytest.mark.parametrize('text, expected', _CALL_CASES)
def test_call(text, expected):
  assert py_flag_parser.parse(text) == expected


# ---- Const refs ----

_CONST_REF_CASES = [
    pytest.param(
        'pathlib.Path',
        konfig.ConfigDict({'__const__': 'pathlib:Path'}),
        id='simple',
    ),
    pytest.param(
        'path.to.project.module.MyEnum.VALUE',
        konfig.ConfigDict({
            '__const__': 'path.to.project.module.MyEnum:VALUE',
        }),
        id='deep',
    ),
    pytest.param(
        'np.int32',
        konfig.ConfigDict({'__const__': 'numpy:int32'}),
        id='alias_np',
    ),
    pytest.param(
        'kd.data.Xxx',
        konfig.ConfigDict({'__const__': 'kauldron.kd:data.Xxx'}),
        id='alias_kd',
    ),
]


@pytest.mark.parametrize('text, expected', _CONST_REF_CASES)
def test_const_ref(text, expected):
  assert py_flag_parser.parse(text) == expected


# ---- Explicit colon qualnames ----

_EXPLICIT_COLON_CASES = [
    pytest.param(
        'path.to.module:MyEnum.VALUE',
        konfig.ConfigDict({
            '__const__': 'path.to.module:MyEnum.VALUE',
        }),
        id='const_ref',
    ),
    pytest.param(
        'path.to.module:MyClass(x=1)',
        konfig.ConfigDict({
            '__qualname__': 'path.to.module:MyClass',
            'x': 1,
        }),
        id='call',
    ),
    pytest.param(
        'path.to.module:MyEnum.VALUE(x=1)',
        konfig.ConfigDict({
            '__qualname__': 'path.to.module:MyEnum.VALUE',
            'x': 1,
        }),
        id='nested_attr_call',
    ),
    pytest.param(
        "xx.yy:aa.bb(value='aa:ff')",
        konfig.ConfigDict({
            '__qualname__': 'xx.yy:aa.bb',
            'value': 'aa:ff',
        }),
        id='colon_in_str_and_qualname',
    ),
]


@pytest.mark.parametrize('text, expected', _EXPLICIT_COLON_CASES)
def test_explicit_colon(text, expected):
  assert py_flag_parser.parse(text) == expected


# ---- Collections ----

_COLLECTION_CASES = [
    pytest.param(
        '[types.SimpleNamespace(), 4, "hello"]',
        [
            konfig.ConfigDict({'__qualname__': 'types:SimpleNamespace'}),
            4,
            'hello',
        ],
        id='list',
    ),
    pytest.param('[]', [], id='empty_list'),
    pytest.param('(1, 2, 3)', (1, 2, 3), id='tuple'),
    pytest.param('(1,)', (1,), id='single_element_tuple'),
    pytest.param('()', (), id='empty_tuple'),
    pytest.param(
        '{"key": types.SimpleNamespace(x=1)}',
        {
            'key': konfig.ConfigDict({
                '__qualname__': 'types:SimpleNamespace',
                'x': 1,
            }),
        },
        id='dict',
    ),
    pytest.param('{}', {}, id='empty_dict'),
    pytest.param(
        '{"key:yy": xx.yy:aa.bb(value="aa:ff")}',
        {
            'key:yy': konfig.ConfigDict({
                '__qualname__': 'xx.yy:aa.bb',
                'value': 'aa:ff',
            }),
        },
        id='complex_dict_with_colons',
    ),
]


@pytest.mark.parametrize('text, expected', _COLLECTION_CASES)
def test_collection(text, expected):
  assert py_flag_parser.parse(text) == expected


# ---- Literals ----

_LITERAL_CASES = [
    pytest.param('42', 42, id='int'),
    pytest.param('3.14', 3.14, id='float'),
    pytest.param('.5', 0.5, id='float_leading_dot'),
    pytest.param('1.', 1.0, id='float_trailing_dot'),
    pytest.param('1e3', 1000.0, id='float_scientific'),
    pytest.param('"hello"', 'hello', id='string_double_quotes'),
    pytest.param("'hello'", 'hello', id='string_single_quotes'),
    pytest.param(
        'my.module.Cls(x="a:b")',
        konfig.ConfigDict({
            '__qualname__': 'my.module:Cls',
            'x': 'a:b',
        }),
        id='string_with_colon',
    ),
    pytest.param("'aa:bb\\'aa:bb'", "aa:bb'aa:bb", id='escaped_quotes'),
    pytest.param(
        "'aa:bb\\'aa:bb\\'aa:bb'",
        "aa:bb'aa:bb'aa:bb",
        id='multiple_escaped_quotes',
    ),
    pytest.param('""', '', id='empty_string_double'),
    pytest.param("''", '', id='empty_string_single'),
    pytest.param('True', True, id='true'),
    pytest.param('False', False, id='false'),
    pytest.param('None', None, id='none'),
    pytest.param('-1', -1, id='negative_int'),
    pytest.param('-3.14', -3.14, id='negative_float'),
    pytest.param('0xff', 255, id='hex'),
    pytest.param('0XFF', 255, id='hex_upper'),
    pytest.param('0o17', 15, id='oct'),
    pytest.param('0b1010', 10, id='bin'),
    pytest.param('1_000', 1000, id='int_underscores'),
    pytest.param('1_000.0', 1000.0, id='float_underscores'),
    pytest.param('2j', 2j, id='imag_int'),
    pytest.param('3.14j', 3.14j, id='imag_float'),
    pytest.param('-0xff', -255, id='negative_hex'),
    pytest.param('-2j', -2j, id='negative_imag'),
]


@pytest.mark.parametrize('text, expected', _LITERAL_CASES)
def test_literal(text, expected):
  assert py_flag_parser.parse(text) == expected


# ---- Error cases ----

_ERROR_CASES = [
    pytest.param('not a valid expression !!!', id='invalid_expression'),
    pytest.param('Name', id='single_name_qualname'),
    pytest.param('Name()', id='single_name_call'),
]


@pytest.mark.parametrize('text', _ERROR_CASES)
def test_error(text):
  with pytest.raises(ValueError, match='Invalid py:: expression'):
    py_flag_parser.parse(text)
