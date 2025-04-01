# Copyright 2024 The kauldron Authors.
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

from typing import TypedDict

import jax.numpy as jnp

from kauldron.ktyping import dtypes
from kauldron.ktyping import typeguard_checkers  # pylint: disable=unused-import
from kauldron.ktyping.array_type_meta import ArrayTypeMeta  # pylint: disable=g-importing-member
from kauldron.ktyping.array_types import Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.ktyping.decorator import typechecked  # pylint: disable=g-importing-member
import numpy as np
import pytest
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

  with pytest.raises(typeguard.TypeCheckError):
    f(jnp.zeros_like(x), y)  # wrong array type

  with pytest.raises(typeguard.TypeCheckError):
    f(x, y.astype(np.int32))  # wrong dtype

  with pytest.raises(typeguard.TypeCheckError):
    f(x, x)  # wrong shape


def test_array_type_check_with_regular_types():
  @typechecked
  def f(x: Float["*b h w"], y: int, z: str) -> Float["*b"]:
    return np.sum(x, axis=(-1, -2)) + len(z) + y

  x = np.zeros((2, 3, 5, 7), dtype=np.float32)
  assert f(x, 7, "abc").shape == (2, 3)

  with pytest.raises(typeguard.TypeCheckError):
    f(x, "abc", 7)  # pytype: disable=wrong-arg-types


def test_array_type_check_with_containers():
  @typechecked
  def f(
      x: list[Float["a b"]], y: dict[str, Int[""]]
  ) -> tuple[Float["a b"], int]:
    return x[0], int(y["a"])

  x = [np.zeros((2, 3), dtype=np.float32)]
  y = {"a": np.array(7, dtype=np.int32)}
  assert f(x, y)[1] == 7

  with pytest.raises(typeguard.TypeCheckError):
    f(x, {"a": "b"})  # wrong type in dict

  with pytest.raises(typeguard.TypeCheckError):
    f([x[0][0]], y)  # wrong shape in list


def test_array_type_union_check():
  @typechecked
  def f(x: Float["*b h w"] | Float["*b"], y: Float["*b"]) -> Float["*b"]:
    del x
    return y

  x = np.zeros((2, 3, 5, 7), dtype=np.float32)
  y = np.zeros((2, 3), dtype=np.float32)
  assert f(x, y).shape == (2, 3)
  assert f(y, y).shape == (2, 3)

  with pytest.raises(typeguard.TypeCheckError):
    f(x[0], y)


def test_fstring_interpolation():
  @typechecked
  def f(x: Float["{batch_size} h {len(text)*5}"], batch_size: int, text: str):
    del x, batch_size, text
    return

  x = np.zeros((8, 5, 15), dtype=np.float32)

  f(x, 8, "abc")

  with pytest.raises(typeguard.TypeCheckError):
    f(x, 4, "abc")

  with pytest.raises(typeguard.TypeCheckError):
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
  assert f(x, y) == 8

  with pytest.raises(typeguard.TypeCheckError):
    f({"a": 7, "b": 42, "c": np.zeros((3,))}, y)  # wrong type in dict

  with pytest.raises(typeguard.TypeCheckError):
    f({"a": 7, "b": np.zeros((2, 3)), "c": np.zeros((2,))}, y)  # wrong shape
