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

from kauldron.ktyping import shape_spec
from kauldron.ktyping import shape_tools
from kauldron.ktyping.array_types import Float  # pylint: disable=g-importing-member
from kauldron.ktyping.decorator import typechecked  # pylint: disable=g-importing-member
import kauldron.ktyping.internal_typing as ktype  # pylint: disable=g-importing-member
import numpy as np
import pytest


@pytest.mark.parametrize(
    "shape_str,expected",
    [
        ("", ()),
        ("*b", (2, 3)),
        ("c", (11,)),
        ("*b h w c", (2, 3, 5, 7, 11)),
        ("2 2+5*6 8//2 5-4+1 10/(3+2) 2**2", (2, 32, 4, 2, 2, 4)),
        ("sum(*b,h) max(w,4) 2*c+h", (10, 7, 27)),
        ("c//2 *b *b h c**2", (5, 2, 3, 2, 3, 5, 121)),
    ],
)
def test_eval_shape_single_alternative(shape_str: str, expected):
  dim_values = ktype.DimValues({"b": (2, 3), "h": (5,), "w": (7,), "c": (11,)})
  s = shape_tools._eval_shape(shape_str, frozenset([dim_values]))
  assert s == expected


def test_eval_shape_multiple_equivalent_alternatives():
  dim_values1 = ktype.DimValues({"a": (2,), "b": (3,), "c": (5,)})
  dim_values2 = ktype.DimValues({"a": (3,), "b": (2,), "c": (5,)})
  alternatives = frozenset([dim_values1, dim_values2])
  assert shape_tools._eval_shape("a*b c", alternatives) == (6, 5)
  assert shape_tools._eval_shape("sum(a,b,c)", alternatives) == (10,)
  assert shape_tools._eval_shape("c 3 5", alternatives) == (5, 3, 5)


def test_eval_shape_no_alternatives():
  assert shape_tools._eval_shape("1 2 3", frozenset()) == (1, 2, 3)
  with pytest.raises(shape_spec.ShapeError, match="No possible shape found"):
    shape_tools._eval_shape("a*b c", frozenset())


def test_eval_shape_unknown_dimension():
  with pytest.raises(shape_spec.ShapeError, match="No value known"):
    shape_tools._eval_shape(
        "a*b d", frozenset([ktype.DimValues({"a": (2,), "b": (35,)})])
    )


def test_eval_shape_ambiguous_alternatives():
  dim_values1 = ktype.DimValues({"a": (2,), "b": (3,), "c": (5,)})
  dim_values2 = ktype.DimValues({"a": (3,), "b": (2,), "c": (5,)})
  alternatives = frozenset([dim_values1, dim_values2])
  with pytest.raises(shape_spec.ShapeError, match="is ambiguous") as exc_info:
    shape_tools._eval_shape("a c", alternatives)
  # check that each alternative is mentioned in the error message
  exc_info.match("(3, 5)")
  exc_info.match("(2, 5)")


def test_shape_works_in_typechecked_func():
  @typechecked
  def f(x: Float["a b"]):
    del x
    return shape_tools.shape("b a")

  assert f(np.zeros((3, 2))) == (2, 3)
