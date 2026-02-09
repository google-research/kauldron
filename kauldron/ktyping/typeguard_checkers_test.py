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
from kauldron.ktyping import errors
from kauldron.ktyping import frame_utils
from kauldron.ktyping import pytree
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
