# Copyright 2025 The kauldron Authors.
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

"""One-line documentation for test_tools."""

import dataclasses
from typing import Literal, Optional, TypedDict, Union

import jax
import jax.numpy as jnp
import jaxtyping as jt
import kauldron.ktyping as kt
from kauldron.ktyping import utils
import kauldron.typing as kdt
import numpy as np
import pytest
import tensorflow as tf


class Foo:
  pass


@pytest.mark.parametrize(
    "obj, expected_name",
    [
        (None, "None"),
        (int, "int"),
        (8, "int"),
        ("abc", "str"),
        (Foo, "Foo"),  # kauldron.ktyping.utils_test.Foo
        (Foo(), "Foo"),  # kauldron.ktyping.utils_test.Foo
        (list[int], "list[int]"),
        (jax.Array, "Array"),
        (Literal["foo"], "Literal['foo']"),
        (int | None, "int | None"),
    ],
)
def test_get_type_name(obj, expected_name):
  assert utils.get_type_name(obj) == expected_name


@pytest.mark.parametrize(
    "obj, expected_name",
    [
        (Foo, "kauldron.ktyping.utils_test.Foo"),
        (Foo(), "kauldron.ktyping.utils_test.Foo"),
        (jax.Array, "jax.Array"),
        (np.zeros((1,)), "numpy.ndarray"),
    ],
)
def test_get_type_name_full_path(obj, expected_name):
  assert utils.get_type_name(obj, full_path=True) == expected_name


@pytest.mark.parametrize(
    "obj, expected_name",
    [
        (1, "1"),
        ("short string", "'short string'"),
        (1.234, "1.234"),
        (True, "True"),
        (None, "None"),
        (jnp.array([1.0, 2.0, 3.0]), "jax.f32[3]"),
        (np.array([[1, 2, 3]], dtype=np.int32), "np.i32[1 3]"),
        (tf.constant([True, False]), "tf.bool_[2]"),
        (
            "this is a very long string that should be truncated",
            "'this is a very long string that sho...'",
        ),
    ],
)
def test_format_value(obj, expected_name):
  assert utils.format_value(obj) == expected_name


@pytest.mark.parametrize(
    "obj, expected_str",
    [
        (np.array([1, 2, 3], dtype=np.int32), "int32"),
        (jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32), "float32"),
        (tf.constant([True, False], dtype=tf.bool), "bool"),
        (np.int32, "np.int32"),
        (np.floating, "np.floating"),
    ],
)
def test_get_dtype_str(obj, expected_str):
  assert utils.get_dtype_str(obj) == expected_str


def test_code_location_from_any():
  def f():
    pass

  class C:
    pass

  @dataclasses.dataclass(kw_only=True)
  class D:
    pass

  class T(TypedDict):
    pass

  f_loc = utils.CodeLocation.from_any(f)
  assert f_loc.file == __file__
  assert f_loc.line == f.__code__.co_firstlineno
  assert f_loc.description == "function 'f'"
  assert f_loc.module_name == __name__

  c_loc = utils.CodeLocation.from_any(C)
  assert c_loc.file == __file__
  assert isinstance(c_loc.line, int)
  assert c_loc.description == "class 'C'"
  assert f_loc.module_name == __name__

  d_loc = utils.CodeLocation.from_any(D)
  assert d_loc.file == __file__
  assert isinstance(d_loc.line, int)
  assert d_loc.description == "dataclass 'D'"
  assert f_loc.module_name == __name__

  t_loc = utils.CodeLocation.from_any(T)
  assert t_loc.file == __file__
  assert isinstance(t_loc.line, int)
  assert t_loc.description == "TypedDict 'T'"
  assert f_loc.module_name == __name__


def test_code_location_as_decorator():
  def check_code_location(f):
    f_loc = utils.CodeLocation.from_any(f)
    assert f_loc.file == __file__
    assert isinstance(f_loc.line, int)
    assert f_loc.module_name == __name__
    return f_loc

  class Test:

    @check_code_location
    def my_method(self):
      pass

    @check_code_location
    @classmethod
    def my_classmethod(cls):
      pass

    @check_code_location
    @staticmethod
    def my_staticmethod():
      pass

  _ = Test()


def test_code_location_from_any_builtins_exec():
  s = utils.CodeLocation.from_any(int)
  assert s.file == "<builtins>"
  assert s.line is None

  # dynamically create a function using exec
  local = {}
  exec("def f(): pass", {}, local)  # pylint: disable=exec-used
  f = local["f"]

  assert utils.CodeLocation.from_any(f).file == "<string>"


@pytest.mark.parametrize(
    "annot",
    [
        # plain jaxtyping types
        jt.Float[jt.Array, ""],
        jt.UInt8[jt.Array, "a b"],
        jt.Scalar,
        jt.ScalarLike,
        jt.PyTree[jt.Float[jt.Array, ""], "t"],
        # composite jaxtyping types
        Union[float, jt.Float32[jt.Array, ""]],
        Optional[jt.Int[jt.Array, ""]],
        int | jt.Integer[jt.Array, ""],
        tuple[int, bool, jt.Bool[jt.Array, ""]],
        dict[str, jt.Complex64[jt.Array, ""]],
        # plain kd.typing types
        kdt.Float[""],
        kdt.UInt8["a b"],
        # composite kd.typing types
        Union[float, kdt.Float32["n"]],
        Optional[kdt.Int[""]],
        int | kdt.Integer[""],
        tuple[int, bool, kdt.Bool[""]],
        dict[str, kdt.Complex64[""]],
    ],
)
def test_contains_jaxtyping_type_true_for_jaxtyping_types(annot):
  assert utils.contains_jaxtyping_type(annot)


@pytest.mark.parametrize(
    "annot",
    [
        # basic types
        int,
        float,
        bool,
        complex,
        # composite types
        tuple[int, bool],
        dict[str, complex],
        # plain ktyping types
        kt.Float[""],
        kt.UInt8["a b"],
        # composite ktyping types
        Union[float, kt.Float32["n"]],
        Optional[kt.Int[""]],
        int | kt.Int[""],
        tuple[int, bool, kt.Bool[""]],
        dict[str, kt.Complex64[""]],
    ],
)
def test_contains_jaxtyping_type_false_otherwise(annot):
  assert not utils.contains_jaxtyping_type(annot)


def test_contains_jaxtyping_type_dataclass():
  @dataclasses.dataclass(kw_only=True)
  class Bar:
    a: int
    x: jt.Float[jt.Array, ""] | None

  assert utils.contains_jaxtyping_type(Bar)


def test_contains_jaxtyping_type_typeddict():
  class Bar(TypedDict):
    a: int
    x: tuple[int, jt.Float[jt.Array, ""]]

  assert utils.contains_jaxtyping_type(Bar)
