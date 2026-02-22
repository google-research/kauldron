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

"""Tests for the @typechecked decorator."""

import dataclasses
import inspect
from typing import TypedDict, Generator

import jax.numpy as jnp
from kauldron.ktyping import dim_view
from kauldron.ktyping import dtypes
from kauldron.ktyping import errors
from kauldron.ktyping import frame_utils
from kauldron.ktyping import scope
from kauldron.ktyping import typeguard_checkers  # pylint: disable=unused-import
from kauldron.ktyping.array_type_meta import ArrayTypeMeta  # pylint: disable=g-importing-member
from kauldron.ktyping.array_types import Float, Int, TfArray, XArray  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.ktyping.decorator import typechecked  # pylint: disable=g-importing-member
import numpy as np
import pytest
import tensorflow as tf
import typeguard


def test_basic():
  @typechecked
  def f(x: int, y: int) -> int:
    return x + y

  assert f(1, 2) == 3
  with pytest.raises(typeguard.TypeCheckError):
    f(1, 2.0)  # pytype: disable=wrong-arg-types


def test_array_type_check():
  NpFloat = ArrayTypeMeta(  # pylint: disable=invalid-name
      "NpFloat", array_types=(np.ndarray,), dtype=dtypes.floating
  )

  @typechecked
  def f(x: NpFloat["*b h w"], y: Float["*b"]) -> Float["*b"]:
    return np.sum(x, axis=(-1, -2)) + y

  x = np.zeros((2, 3, 5, 7), dtype=np.float32)
  y = np.zeros((2, 3), dtype=np.float32)
  assert f(x, y).shape == (2, 3)

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    f(jnp.zeros_like(x), y)

  with pytest.raises(
      errors.KTypeCheckError, match="is not dtype-compatible with np.floating"
  ):
    f(x, y.astype(np.int32))

  with pytest.raises(
      errors.KTypeCheckError, match="is not shape-compatible with '\\*b'"
  ):
    f(x, x)


def test_array_type_check_with_regular_types():
  @typechecked
  def f(x: Float["*b h w"], y: int, z: str) -> Float["*b"]:
    return np.sum(x, axis=(-1, -2)) + len(z) + y

  x = np.zeros((2, 3, 5, 7), dtype=np.float32)
  assert f(x, 7, "abc").shape == (2, 3)

  with pytest.raises(typeguard.TypeCheckError):
    f(x, "abc", 7)  # pytype: disable=wrong-arg-types

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    f("array", 7, "abc")  # pytype: disable=wrong-arg-types


def test_array_type_check_with_containers():
  @typechecked
  def f(
      x: list[Float["a b"]], y: dict[str, Int[""]]
  ) -> tuple[Float["a b"], int]:
    return x[0], int(y["a"])

  x = [np.zeros((2, 3), dtype=np.float32)]
  y = {"a": np.array(7, dtype=np.int32)}
  assert f(x, y)[1] == 7

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    f(x, {"a": "b"})  # wrong type in dict

  with pytest.raises(
      errors.KTypeCheckError, match="is not shape-compatible with"
  ):
    f([x[0][0]], y)  # wrong shape in list


def test_simple_array_type_union_check():
  @typechecked
  def f(x: Float["*b h w"] | Float["*b"], y: Float["*b"]) -> Float["*b"]:
    del x
    return y

  x = np.zeros((2, 3, 5, 7), dtype=np.float32)
  y = np.zeros((2, 3), dtype=np.float32)
  assert f(x, y).shape == (2, 3)
  assert f(y, y).shape == (2, 3)

  with pytest.raises(
      errors.KTypeCheckError, match="is not shape-compatible with '\\*b'"
  ):
    f(x[0], y)


def test_non_greedy_array_type_union_check():
  @typechecked
  def f(x: Float["a"] | Float["b"], y: Float["a"]):
    del x, y
    return dim_view.dim["b"]

  # greedy matching would pick the first alternative and fail
  x = np.zeros((5,), dtype=np.float32)
  y = np.zeros((3,), dtype=np.float32)
  assert f(x, y) == 5


def test_compound_array_type_union_check():
  @typechecked
  def f(x: Float["3"] | Int["1"]) -> Float["*b"]:
    return x

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    f(tf.zeros((3), dtype=np.float32))  # does not match any array type

  with pytest.raises(
      errors.KTypeCheckError, match="is not dtype-compatible with"
  ):
    f(np.zeros((3), dtype=np.bool_))  # does not match any dtype

  with pytest.raises(
      errors.KTypeCheckError, match="is not shape-compatible with"
  ):
    f(np.zeros((5), dtype=np.float32))  # does not match any shape

  with pytest.raises(
      errors.KTypeCheckError,
      match="did not match any of its annotations due to a combination",
  ):
    f(np.zeros((3,), dtype=np.int32))  # fails on combination of dtype and shape


def test_fstring_interpolation():
  @typechecked
  def f(x: Float["{batch_size} h {len(text)*5}"], batch_size: int, text: str):
    del x, batch_size, text
    return

  x = np.zeros((8, 5, 15), dtype=np.float32)

  f(x, 8, "abc")

  with pytest.raises(
      errors.KTypeCheckError, match="is not shape-compatible with"
  ):
    f(x, 4, "abc")

  with pytest.raises(
      errors.KTypeCheckError, match="is not shape-compatible with"
  ):
    f(x, 8, "abcdefg")


def test_typeddict_check():
  class Custom(TypedDict):
    a: int
    b: Float["*b c"]
    c: Float["c"]

  @typechecked
  def f(x: Custom, y: Int["*b"]) -> int:
    return x["a"] + y.ndim

  x = {"a": 7, "b": np.zeros((2, 3)), "c": np.zeros((3,))}
  y = np.zeros((7,), dtype=np.int32)
  # Note that the *b from Custom is not shared with the *b from argument y.
  assert f(x, y) == 8

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    f({"a": 7, "b": 42, "c": np.zeros((3,))}, y)  # wrong type in dict

  with pytest.raises(
      errors.KTypeCheckError, match="is not shape-compatible with"
  ):
    f({"a": 7, "b": np.zeros((2, 3)), "c": np.zeros((2,))}, y)  # wrong shape


def test_transparent_typechecked_context():
  # get linenumber of this function
  lineno = inspect.getsourcelines(test_transparent_typechecked_context)[1]

  @typechecked(new_scope=False)
  def f():
    sscope = scope.get_current_scope()
    assert isinstance(sscope, scope.TransparentScope)
    # The scope is transparent but the source should still point to f()
    assert sscope.source.description == "function 'f'"
    assert sscope.source.file.endswith("decorator_test.py")
    assert sscope.source.line == lineno + 4  # + 4 for the '@typechecked' line

    # can access the dim assignments from the parent scope
    assert dim_view.dim["h"] == 7

  with typechecked():
    dim_view.dim["h"] = 7
    f()


def test_typechecked_context():
  lineno_of_this_fn = inspect.getsourcelines(test_typechecked_context)[1]
  with typechecked():
    sscope = scope.get_current_scope()
    assert sscope.source.description == "typechecked context"
    assert sscope.source.file.endswith("decorator_test.py")
    assert sscope.source.line == lineno_of_this_fn + 2  # +2 for the 'with' line
    typeguard_checkers.check_type(np.zeros((2, 3)), Float["h w"])
    assert dim_view.dim["h"] == 2
    assert dim_view.dim["w"] == 3


@pytest.mark.parametrize("new_scope", [True, False])
def test_nested_typechecked_context(new_scope: bool):
  with typechecked():
    sscope1 = scope.get_current_scope()
    with typechecked(new_scope=new_scope):
      sscope2 = scope.get_current_scope()
      assert sscope1 != sscope2

    sscope3 = scope.get_current_scope()
    assert sscope1 == sscope3

  with pytest.raises(frame_utils.NoActiveScopeError):
    scope.get_current_scope()


def test_typechecked_dataclass_init():
  @typechecked
  @dataclasses.dataclass
  class Foo:
    x: int
    y: Float["*b"]
    z: str = "abc"

    def __post_init__(self):
      self.x = int(self.x)

  _ = Foo(x=7.2, y=np.zeros((2, 3), dtype=np.float32))  # pytype: disable=wrong-arg-types

  with pytest.raises(errors.KTypeCheckError, match="is not dtype-compatible"):
    Foo(x=7, y=np.zeros((2, 3), dtype=np.int32))  # wrong dtype


def test_typechecked_dataclass_init_with_subclass():
  @typechecked  # should not trigger when instantiating Bar
  @dataclasses.dataclass(frozen=True)
  class Foo:
    x: int

  @typechecked
  @dataclasses.dataclass(frozen=True)
  class Bar(Foo):
    x: float

    def __init__(self, x):  # pylint: disable=useless-parent-delegation
      super().__init__(x)  # pytype: disable=wrong-arg-types

  _ = Bar(x=7.2)


def test_typechecked_dataclass_arguments():
  @dataclasses.dataclass
  class Unchecked:
    a: int
    b: Float["*b"]

  @typechecked
  @dataclasses.dataclass
  class Checked:
    a: int
    b: Float["*b"]

  @typechecked
  def f(x: Unchecked, y: Checked):
    return x.a, y.a

  x = Unchecked(a="seven", b=np.zeros((2, 3), dtype=np.int32))  # pytype: disable=wrong-arg-types
  y = Checked(a=8, b=np.zeros((2, 3), dtype=np.float32))

  # This call is ok, because the Unchecked dataclass is not typechecked.
  assert f(x, y) == ("seven", 8)

  y.a = "eight"  # pytype: disable=annotation-type-mismatch

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    f(x, y)  # y.a is not an int

  y.a = 8
  y.b = np.zeros((2, 3), dtype=np.int32)

  with pytest.raises(
      errors.KTypeCheckError, match="is not dtype-compatible with"
  ):
    f(x, y)  # y.b is not dtype-compatible with np.floating


def test_typechecked_method():
  class Foo:

    @typechecked
    def bar(self, x: int) -> int:
      return x * 2

  f = Foo()
  f.bar(1)  # ok

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    f.bar("one")  # pytype: disable=wrong-arg-types


def test_typechecked_classmethod():
  class Foo:

    @typechecked
    @classmethod
    def bar(cls, x: int) -> int:
      return x * 2

  Foo.bar(1)  # ok

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    Foo.bar("one")  # pytype: disable=wrong-arg-types


def test_typechecked_property_success():
  class Foo:

    @typechecked
    @property
    def bar(self) -> int:
      return 42

    @typechecked
    @bar.setter
    def bar(self, value: int):
      self._bar = value

    @typechecked
    @bar.deleter
    def bar(self) -> None:
      pass

  assert Foo().bar == 42
  Foo().bar = 17
  del Foo().bar


def test_typechecked_property_fail():
  class Foo:

    @typechecked
    @property
    def bar(self) -> int:
      return "42"  # pytype: disable=bad-return-type

    @typechecked
    @bar.setter
    def bar(self, value: int):
      self._bar = value

    @typechecked
    @bar.deleter
    def bar(self) -> None:
      return 42  # pytype: disable=bad-return-type

  with pytest.raises(errors.KTypeCheckError, match="property 'bar'"):
    _ = Foo().bar

  with pytest.raises(errors.KTypeCheckError, match="property 'bar'"):
    Foo().bar = "None"

  with pytest.raises(errors.KTypeCheckError, match="property 'bar'"):
    del Foo().bar


def test_typechecked_generator_args():
  @typechecked
  def my_gen(a: int) -> Generator[str, None, bool]:
    yield "a"
    return True if a != 3 else "False"  # pytype: disable=bad-return-type

  assert [x for x in my_gen(1)] == ["a"]

  with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
    _ = [x for x in my_gen("one")]  # pytype: disable=wrong-arg-types

  with pytest.raises(errors.KTypeCheckError, match="return value"):
    _ = [x for x in my_gen(3)]
    assert False


def test_typechecked_with_tf_tensor_eager():
  @typechecked
  def fn(x: TfArray["B H W C"]) -> TfArray["B H W C"]:
    return x

  t = tf.constant(np.zeros((2, 5, 5, 3), dtype=np.int32))
  result = fn(t)
  assert result.shape == (2, 5, 5, 3)


def test_typechecked_with_tf_tensor_graph():
  @typechecked
  def fn(x: TfArray["B H W C"]) -> TfArray["B H W C"]:
    return x

  with tf.Graph().as_default():
    t = tf.reshape(tf.range(150), (2, 5, 5, 3))
    result = fn(t)
    assert result.shape == (2, 5, 5, 3)


def test_typechecked_with_xarray_tf_graph():
  @typechecked
  def fn(x: XArray["*b T H W C"]) -> XArray["*b T H W C"]:
    return x

  with tf.Graph().as_default():
    t = tf.cast(tf.reshape(tf.range(3), [3, 1, 1, 1]), tf.float32)
    result = fn(t)
    assert result.shape == (3, 1, 1, 1)


def test_typechecked_with_unknown_tf_dims_graph():
  @typechecked
  def fn(x: TfArray["B H W C"]) -> TfArray["B H W C"]:
    return x

  with tf.Graph().as_default():
    t = tf.compat.v1.placeholder(tf.float32, shape=[None, 5, 5, 3])
    result = fn(t)
    assert result.shape[1] == 5
