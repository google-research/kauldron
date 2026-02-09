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

import jaxtyping as jt
from kauldron.typing import enable_kd_type_checking  # pylint: disable=g-importing-member
from kauldron.typing import Float, Shape, TypeCheckError, typechecked  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.typing import shape_spec  # pylint: disable=g-bad-import-order
from kauldron.typing import type_check  # pylint: disable=g-bad-import-order

import numpy as np
import pytest

enable_kd_type_checking()


def test_decorator():
  @typechecked
  def _foo(_: Float["a"]) -> Float["a"]:
    return np.zeros((1,))

  _foo(np.zeros((1,)))

  with pytest.raises(TypeCheckError):
    _foo(np.zeros((1, 2)))

  with pytest.raises(TypeCheckError):
    _foo(np.zeros((8,)))


def test_decorator2():
  @typechecked
  def _foo(x: Float["a 1"], y: Float["a b"]) -> Float["a/2 b+1"]:  # pylint: disable=unused-argument
    return np.zeros((5, 4))

  _foo(np.zeros((10, 1)), np.zeros((10, 3)))

  with pytest.raises(TypeCheckError):
    _foo(x=np.zeros((10, 1)), y=np.zeros((8, 3)))

  with pytest.raises(TypeCheckError):
    _foo(x=np.zeros((6, 1)), y=np.zeros((6, 3)))

  with pytest.raises(TypeCheckError):
    _foo(x=np.zeros((10, 2)), y=np.zeros((10, 3)))


def _dataclass_test_helper(cls):
  # The following function has an intentional bug.
  # It returns a Float["B T"] instead of Float["T B"].
  @typechecked
  def _foo(a: cls) -> Float["B T"]:
    return a.a

  # Technically this should is still wrong, because although the size of the
  # dimensions match (B = 2, T = 2) gets mapped to (T = 2, B = 2), the order
  # of the dimensions is swapped.
  # But the type checker can't see that.
  _foo(cls(a=np.zeros((2, 2)), b=np.zeros((2, 2))))

  with pytest.raises(TypeCheckError):
    # Wrong type! np.ndarray instead of A.
    _foo(np.zeros((2, 2)))

  with pytest.raises(TypeCheckError):
    # B,T are not interchangeable here, hence this will fail.
    _foo(cls(a=np.zeros((2, 3)), b=np.zeros((2, 3))))


def test_dataclass():

  @dataclasses.dataclass
  class A:
    a: Float["T B"]

  @dataclasses.dataclass
  class B(A):
    b: Float["T B"]

  @dataclasses.dataclass
  class C:
    a: Float["T B"]
    b: Float["T B"]

  _dataclass_test_helper(B)
  _dataclass_test_helper(C)


@dataclasses.dataclass
class TestB:
  a: "TestA"


# Note: These datac-classes need to be defined outside the test_dataclass()
# otherwise one cannot annotate TestB.a with the type TestA, because the type
# checker will get confused.
@dataclasses.dataclass
class TestA:
  a: Float["T B"]


def test_nested_dataclass():

  @typechecked
  def _foo(b: TestB) -> TestA:
    return b.a

  _foo(TestB(a=TestA(a=np.zeros((2, 2)))))

  with pytest.raises(TypeCheckError):
    # Wrong shape
    _foo(TestB(a=TestA(a=np.zeros((2, 2, 2)))))


@dataclasses.dataclass
class NestedA:
  x: int
  y: dict[str, bool]
  z: "NestedA"  # Recursive dataclass has to be defined in the global scope.


def test_union_type():

  assert not type_check._is_kd_dataclass(NestedA)

  @dataclasses.dataclass
  class A:
    x: int
    y: jt.Float[jt.Array, "T B"]  # jaxtyping is not a Kauldron dataclass

  assert not type_check._is_kd_dataclass(A)

  @dataclasses.dataclass
  class B:
    x: int
    y: Float["T B"]

  assert type_check._is_kd_dataclass(B)


@typechecked
def test_shape_fails_without_set_shape():
  with pytest.raises(shape_spec.ShapeError):
    Shape("B")


@typechecked
def test_set_shape_with_int():
  type_check.set_shape("B", 4)
  assert Shape("B") == (4,)


@typechecked
def test_set_shape_with_sequence_single():
  type_check.set_shape("B", [4])
  assert Shape("B") == (4,)


@typechecked
def test_set_shape_with_sequence_multiple():
  type_check.set_shape("H W", [224, 224])
  assert Shape("H W") == (224, 224)
