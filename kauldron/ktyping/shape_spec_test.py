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

"""Tests for shape_spec."""

from kauldron.ktyping import shape_spec
from kauldron.ktyping import shape_spec_parser
from kauldron.ktyping.constraints import Constraints  # pylint: disable=g-importing-member
from kauldron.ktyping.internal_typing import Shape  # pylint: disable=g-importing-member
import pytest


_empty = frozenset([Constraints()])


@pytest.mark.parametrize(
    "shape_str, expected",
    [
        ("", ()),
        ("*b", (8, 16)),
        ("c", (5,)),
        ("*b h w c", (8, 16, 32, 64, 5)),
        ("2 2+5*6 8//2 5-4+1 10/(3+2) 2**2", (2, 32, 4, 2, 2, 4)),
        ("sum(*b,h) min(w,4) 2*-c+h", (56, 4, 22)),
        ("c//2 *b *b h c**2", (2, 8, 16, 8, 16, 32, 25)),
    ],
)
def test_shape_eval(shape_str: str, expected: Shape):
  constraints = Constraints({"b": (8, 16), "h": (32,), "w": (64,), "c": (5,)})
  spec = shape_spec_parser.parse(shape_str)
  assert spec.evaluate(constraints) == expected


@pytest.mark.parametrize(
    "shape_str,error_msg",
    [
        ("a", "No value known"),
        ("*b #3", "broadcastable"),
        ("h #c", "broadcastable"),
        ("b c", "length"),
        ("h w _c h", "anonymous"),
    ],
)
def test_shape_eval_error(shape_str: str, error_msg: str):
  constraints = Constraints({"b": (8, 16), "h": (32,), "w": (64,), "c": (5,)})
  spec = shape_spec_parser.parse(shape_str)
  with pytest.raises(shape_spec.ShapeError, match=error_msg):
    spec.evaluate(constraints)


@pytest.mark.parametrize(
    "shape_str",
    [
        "1",
        "a b c",
        "2*a+b c c d-1",
        "*b a",
    ],
)
def test_simple_shape_eval_match_roundtrip(shape_str: str):
  constraints = Constraints({"a": (1,), "b": (2,), "c": (3,), "d": (5,)})
  alternatives = frozenset([constraints])
  spec = shape_spec_parser.parse(shape_str)
  shape = spec.evaluate(constraints)
  assert spec.match(shape, alternatives) == alternatives


def test_match_infer_constraints():
  constraints = Constraints({"a": (1,), "b": (2, 4, 6), "c": (3,)})
  alternatives = frozenset([constraints])
  spec = shape_spec_parser.parse("a *b c")
  shape = spec.evaluate(constraints)
  assert spec.match(shape, _empty) == alternatives


def test_match_multiple_vardims():
  spec = shape_spec_parser.parse("a *b c *d a")
  shape = (1, 2, 3, 3, 4, 1)
  assert spec.match(shape, _empty) == {
      Constraints({"a": (1,), "b": (), "c": (2,), "d": (3, 3, 4)}),
      Constraints({"a": (1,), "b": (2,), "c": (3,), "d": (3, 4)}),
      Constraints({"a": (1,), "b": (2, 3), "c": (3,), "d": (4,)}),
      Constraints({"a": (1,), "b": (2, 3, 3), "c": (4,), "d": ()}),
  }


def test_match_optional_dim():
  spec = shape_spec_parser.parse("a? b c? b b")
  assert spec.match((1, 1, 1), _empty) == {Constraints({"b": (1,)})}
  assert spec.match((2, 1, 1, 1), _empty) == {
      Constraints({"a": (2,), "b": (1,)})
  }
  assert spec.match((1, 2, 1, 1), _empty) == {
      Constraints({"c": (2,), "b": (1,)})
  }
  assert spec.match((3, 1, 2, 1, 1), _empty) == {
      Constraints({"a": (3,), "c": (2,), "b": (1,)})
  }
