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
import typing
from typing import Annotated, Never, NoReturn, Optional, Union
from kauldron.ktyping import dim_view
from kauldron.ktyping import errors
from kauldron.ktyping import frame_utils
from kauldron.ktyping import pytree
from kauldron.ktyping import scope as scope_mod
from kauldron.ktyping import typeguard_checkers as tgc
from kauldron.ktyping.array_types import Float, Int, Scalar, ScalarInt  # pylint: disable=g-multiple-import
from kauldron.ktyping.decorator import typechecked  # pylint: disable=g-importing-member
from kauldron.ktyping.typeguard_checkers import check_type  # pylint: disable=g-importing-member
import numpy as np
import pytest
import typeguard


def test_contains_any_array_type():
  assert tgc._contains_any_array_type(Float["*b"])
  assert tgc._contains_any_array_type(Float | int)
  assert tgc._contains_any_array_type(int | Int["*b"])
  assert tgc._contains_any_array_type(Union[int | Int["*b"] | str])

  assert tgc._contains_any_array_type(list[Float["*b"]])
  assert tgc._contains_any_array_type(tuple[str, Float["*b"], int])
  assert tgc._contains_any_array_type(dict[str, Float["*b"]])
  assert tgc._contains_any_array_type(Annotated[Float["*b"], "a b c"])

  assert tgc._contains_any_array_type(tuple[list[dict[str, Float]], ...])

  assert not tgc._contains_any_array_type(int)
  assert not tgc._contains_any_array_type(Union[int, str])
  assert not tgc._contains_any_array_type(int | None)
  assert not tgc._contains_any_array_type(list[str])
  assert not tgc._contains_any_array_type(tuple[str, int, None])
  assert not tgc._contains_any_array_type(dict[str, float])


def test_contains_any_array_type_typeddict():
  class TypedDict(typing.TypedDict):
    x: Float["*b"]
    y: int

  class TypedDict2(typing.TypedDict):
    x: float
    y: int

  assert tgc._contains_any_array_type(TypedDict)
  assert not tgc._contains_any_array_type(TypedDict2)


def test_contains_any_array_type_dataclass():
  @dataclasses.dataclass
  class CustomDataclass1:
    x: tuple[Float["*b"], float]
    y: int

  @dataclasses.dataclass
  class CustomDataclass2:
    x: float
    y: list[int] | dict[str, int]

  assert tgc._contains_any_array_type(CustomDataclass1)
  assert not tgc._contains_any_array_type(CustomDataclass2)


def test_asssert_not_noreturn():
  def f() -> None:
    pass

  tgc.assert_not_noreturn(f, typing.get_type_hints(f)["return"])

  def g() -> NoReturn:
    pass  # pytype: disable=bad-return-type

  with pytest.raises(typeguard.TypeCheckError):
    tgc.assert_not_noreturn(g, typing.get_type_hints(g)["return"])

  def h() -> Never:
    pass  # pytype: disable=bad-return-type

  with pytest.raises(typeguard.TypeCheckError):
    tgc.assert_not_noreturn(h, typing.get_type_hints(h)["return"])


def test_assert_not_never():
  def f(x: int, y: float, z: None) -> None:
    del x, y, z

  for annot in typing.get_type_hints(f).values():
    tgc.assert_not_never(f, annot)

  def g(x: Never):
    del x

  with pytest.raises(typeguard.TypeCheckError):
    for annot in typing.get_type_hints(g).values():
      tgc.assert_not_never(g, annot)


def test_union_with_non_array_types():
  with typechecked():
    array = np.zeros((2, 3), dtype=np.float32)
    # check that no errors are raised for correct types
    check_type(12, Float["*b"] | int)
    check_type(array, Float["*b"] | int)
    check_type(None, Optional[Float["*b"]])
    check_type(array, Optional[Float["*b"]])

    with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
      check_type(12, Float["*b"] | str)
    with pytest.raises(errors.KTypeCheckError, match="is not dtype-compatible"):
      check_type(array, Optional[Int["*b"]])


def test_scalar_types():
  with typechecked():
    check_type(np.array(1), Int[""])
    check_type(np.int32(2), Int[""])
    check_type(np.int32(3), Scalar)
    check_type(np.int32(4), ScalarInt)
    check_type(5, Scalar)
    check_type(6, ScalarInt)

    with pytest.raises(errors.KTypeCheckError, match="is not dtype-compatible"):
      check_type(np.int32(7), Float[""])
    with pytest.raises(errors.KTypeCheckError, match="is not dtype-compatible"):
      check_type(8.9, ScalarInt)


def test_pytree_type_checker():
  with typechecked():
    check_type({"a": 1, "b": (3, 4)}, pytree.PyTree[int])

    with pytest.raises(errors.KTypeCheckError, match="is not an instance of"):
      check_type({"a": 1, "b": (3, "4")}, pytree.PyTree[int])


def test_check_type_in_typechecked_function():
  @typechecked
  def f(x: Float["a"]):
    check_type(x, Float["a"])

  x = np.zeros((7,))
  assert f(x) is None  # no error


def test_simple_types():
  assert tgc.isinstance_(1, int)
  assert not tgc.isinstance_(1, str)
  assert tgc.isinstance_(1, (str, int))
  assert tgc.isinstance_(1, int | str)
  assert tgc.isinstance_(None, Optional[int])
  assert not tgc.isinstance_(1.0, int)


def test_composite_types():
  assert tgc.isinstance_([1, 2], list[int])
  assert not tgc.isinstance_(["1", "2"], list[int])
  assert tgc.isinstance_({"a": 1}, dict[str, int])
  assert not tgc.isinstance_({"a": "1"}, dict[str, int])


def test_array_types_with_scope():
  x = np.zeros((2, 3), dtype=np.float32)
  y = np.ones(4, dtype=np.int32)
  with typechecked():
    assert tgc.isinstance_(x, Float["2 3"])
    assert tgc.isinstance_(x, Float["a b"])
    assert not tgc.isinstance_(x, Float["2 4"])
    assert not tgc.isinstance_(x, Int["2 3"])
    assert tgc.isinstance_(y, Int["4"])
    assert tgc.isinstance_(y, Int["c"])


def test_array_types_no_scope_fails():
  x = np.zeros((2, 3), dtype=np.float32)
  with pytest.raises(frame_utils.NoActiveScopeError):
    tgc.isinstance_(x, Float["2 3"])


def test_mixed_array_and_simple_types_with_scope():
  x = np.zeros((2, 3), dtype=np.float32)
  with typechecked():
    assert tgc.isinstance_(x, Float["2 3"] | int)
    assert tgc.isinstance_(12, Float["2 3"] | int)
    assert not tgc.isinstance_("a", Float["2 3"] | int)
    assert tgc.isinstance_([x], list[Float["2 3"]])
    assert not tgc.isinstance_([12], list[Float["2 3"]])


def test_mixed_array_and_simple_types_no_scope_fails():
  x = np.zeros((2, 3), dtype=np.float32)
  with pytest.raises(frame_utils.NoActiveScopeError):
    tgc.isinstance_(x, Float["2 3"] | int)
  with pytest.raises(frame_utils.NoActiveScopeError):
    tgc.isinstance_(12, Float["2 3"] | int)


def test_isinstance_tuple_with_scope():
  x = np.zeros((2, 3), dtype=np.float32)
  with typechecked():
    assert tgc.isinstance_(x, (Float["2 3"], int))
    assert tgc.isinstance_(12, (Float["2 3"], int))
    assert not tgc.isinstance_("a", (Float["2 3"], int))


def test_isinstance_tuple_no_scope_fails():
  x = np.zeros((2, 3), dtype=np.float32)
  with pytest.raises(frame_utils.NoActiveScopeError):
    tgc.isinstance_(x, (Float["2 3"], int))
  with pytest.raises(frame_utils.NoActiveScopeError):
    tgc.isinstance_(12, (Float["2 3"], int))


def test_check_type_fails_without_scope():
  def unscoped(x):
    check_type(x, Float["a"])

  @typechecked
  def scoped_fn(x: Float["a"]):
    unscoped(x)

  x = np.zeros((7,))
  assert isinstance(x, Float["a"])  # isinstance is okay without a scope

  with pytest.raises(frame_utils.NoActiveScopeError):
    check_type(x, Float["a"])

  with pytest.raises(frame_utils.NoActiveScopeError):
    scoped_fn(x)


# MARK: Phase 0 - Regression tests for $ structure keys in candidates


def test_shape_checking_with_structure_in_scope():
  x = np.zeros((2, 3), dtype=np.float32)
  with typechecked():
    check_type(x, Float["a b"])
    s = scope_mod.get_current_scope(nested_ok=True)
    s.candidates = [dict(c) | {"$S": "fake_treedef"} for c in s.candidates]
    y = np.ones((2, 3), dtype=np.float32)
    check_type(y, Float["a b"])


def test_dim_view_str_ignores_structures():
  x = np.zeros((2, 3), dtype=np.float32)
  with typechecked():
    check_type(x, Float["a b"])
    s = scope_mod.get_current_scope(nested_ok=True)
    s.candidates = [dict(c) | {"$S": "fake_treedef"} for c in s.candidates]
    dv = dim_view.DimView(s)
    dims_str = str(dv)
    assert "$S" not in dims_str
    assert "a" in dims_str


def test_error_display_with_structures_does_not_crash():
  x = np.zeros((2, 3), dtype=np.float32)
  with typechecked():
    check_type(x, Float["a b"])
    s = scope_mod.get_current_scope(nested_ok=True)
    s.candidates = [dict(c) | {"$T": "fake_treedef"} for c in s.candidates]
    with pytest.raises(errors.KTypeCheckError) as exc_info:
      check_type(np.ones((5,), dtype=np.int32), Float["a b"])
    error_str = str(exc_info.value)
    assert "$T" in error_str
    assert "Tree Structures" in error_str


# MARK: Phase 2 - PyTree structure binding tests


def test_pytree_structure_binding():
  with typechecked():
    check_type({"a": 1, "b": (3, 4)}, pytree.PyTree[int, "$S"])


def test_pytree_structure_same_match():
  with typechecked():
    tree1 = {"a": 1, "b": 2}
    tree2 = {"a": 3, "b": 4}
    check_type(tree1, pytree.PyTree[int, "$S"])
    check_type(tree2, pytree.PyTree[int, "$S"])


def test_pytree_structure_mismatch():
  with typechecked():
    check_type({"a": 1, "b": 2}, pytree.PyTree[int, "$S"])
    with pytest.raises(errors.KTypeCheckError):
      check_type([1, 2, 3], pytree.PyTree[int, "$S"])


def test_pytree_structure_different_names():
  with typechecked():
    check_type({"a": 1, "b": 2}, pytree.PyTree[int, "$S"])
    check_type([1, 2, 3], pytree.PyTree[int, "$T"])


def test_pytree_structure_with_arrays():
  with typechecked():
    tree = {"x": np.zeros((3, 4), dtype=np.float32)}
    check_type(tree, pytree.PyTree[Float["b n"], "$S"])


def test_pytree_structure_in_typechecked_fn():
  @typechecked
  def f(x: pytree.PyTree[int, "$S"]) -> pytree.PyTree[int, "$S"]:
    return x

  assert f({"a": 1, "b": 2}) is not None
  assert f({"a": 1, "b": 2}) is not None

  @typechecked
  def g(
      x: pytree.PyTree[int, "$S"],
  ) -> pytree.PyTree[int, "$S"]:
    del x
    return [1, 2, 3]

  with pytest.raises(errors.KTypeCheckError):
    g({"a": 1, "b": 2})
