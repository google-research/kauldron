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

"""Tests."""

import dataclasses
from typing import Annotated, Optional, TypeVar

from kauldron.kontext import type_utils
import pytest

_token_a = object()
_token_b = object()

_T = TypeVar('_T')
_KeyA = Annotated[_T, _token_a]  # pytype: disable=invalid-typevar
_KeyB = Annotated[_T, _token_b]  # pytype: disable=invalid-typevar


class A:
  x0: Annotated[int, _token_a, _token_b]
  _: dataclasses.KW_ONLY
  x: _KeyA[_KeyB[int]]  # pytype: disable=unsupported-operands
  y: _KeyA[float]  # pytype: disable=unsupported-operands
  z: _KeyA[int]  # pytype: disable=unsupported-operands
  z_opt: Optional[_KeyA[int]]  # pytype: disable=unsupported-operands
  z_opt2: None | _KeyA[int]  # pytype: disable=unsupported-operands
  a: int


@pytest.mark.parametrize('tokens', ((_token_a, _token_b), (_KeyA, _KeyB)))
def test_annotated(tokens):
  assert type_utils.get_annotated(A, tokens[0]) == [
      'x0',
      'x',
      'y',
      'z',
      'z_opt',
      'z_opt2',
  ]
  assert type_utils.get_annotated(A, tokens[1]) == ['x0', 'x']


def test_is_optional():
  optional_fields = type_utils.get_optional_fields(A)
  assert optional_fields == ['z_opt', 'z_opt2']
