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

"""Tests for typeguard_checkers.py."""

import typing
from typing import Annotated, Never, NoReturn, Union
from kauldron.ktyping import typeguard_checkers as tgc
from kauldron.ktyping.array_types import Float, Int  # pylint: disable=g-multiple-import
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
