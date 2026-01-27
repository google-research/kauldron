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

from kauldron.ktyping import internal_typing as ktype
from kauldron.ktyping import shape_spec
from kauldron.ktyping import shape_spec_parser
import pytest


_empty = frozenset([ktype.DimValues()])
UNKNOWN_DIM = shape_spec.UNKNOWN_DIM


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
def test_shape_eval(shape_str: str, expected: ktype.Shape):
  dim_values = ktype.DimValues(
      {"b": (8, 16), "h": (32,), "w": (64,), "c": (5,)}
  )
  spec = shape_spec_parser.parse(shape_str)
  assert spec.evaluate(dim_values) == expected


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
  dim_values = ktype.DimValues(
      {"b": (8, 16), "h": (32,), "w": (64,), "c": (5,)}
  )
  spec = shape_spec_parser.parse(shape_str)
  with pytest.raises(shape_spec.ShapeError, match=error_msg):
    spec.evaluate(dim_values)


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
  dim_values = ktype.DimValues({"a": (1,), "b": (2,), "c": (3,), "d": (5,)})
  alternatives = frozenset([dim_values])
  spec = shape_spec_parser.parse(shape_str)
  shape = spec.evaluate(dim_values)
  assert spec.match(shape, alternatives) == alternatives


def test_match_infer_dim_values():
  dim_values = ktype.DimValues({"a": (1,), "b": (2, 4, 6), "c": (3,)})
  alternatives = frozenset([dim_values])
  spec = shape_spec_parser.parse("a *b c")
  shape = spec.evaluate(dim_values)
  assert spec.match(shape, _empty) == alternatives


def test_match_multiple_vardims():
  spec = shape_spec_parser.parse("a *b c *d a")
  shape = (1, 2, 3, 3, 4, 1)
  assert spec.match(shape, _empty) == {
      ktype.DimValues({"a": (1,), "b": (), "c": (2,), "d": (3, 3, 4)}),
      ktype.DimValues({"a": (1,), "b": (2,), "c": (3,), "d": (3, 4)}),
      ktype.DimValues({"a": (1,), "b": (2, 3), "c": (3,), "d": (4,)}),
      ktype.DimValues({"a": (1,), "b": (2, 3, 3), "c": (4,), "d": ()}),
  }


def test_match_known_single_broadcastable_int_dim():
  spec = shape_spec_parser.parse("#5 c")
  candidates = frozenset([ktype.DimValues({"c": (3,)})])
  # when matching broadcastable dims against known values,
  # either 1 or the known value are accepted, and the candidates are unmodified.
  assert spec.match((5, 3), candidates) == candidates
  assert spec.match((1, 3), candidates) == candidates


def test_match_known_broadcastable():
  spec = shape_spec_parser.parse("*#b #c")
  dim_values = ktype.DimValues({"b": (7, 11), "c": (3,)})
  candidates = frozenset([dim_values])
  # when matching broadcastable dims against known values,
  # either 1 or the known value are accepted, and the candidates are unmodified.
  assert spec.match((7, 11, 1), candidates) == candidates
  assert spec.match((1, 11, 3), candidates) == candidates
  assert spec.match((1, 1, 3), candidates) == candidates
  assert spec.match((7, 1, 1), candidates) == candidates


def test_match_unknown_broadcastable():
  spec = shape_spec_parser.parse("*#b #c")
  dim_values = ktype.DimValues({"b": (7, UNKNOWN_DIM), "c": (UNKNOWN_DIM,)})
  candidates = frozenset([dim_values])
  # When matching broadcastable dims against unknown values,
  # the candidates are updated with new information.
  assert spec.match((7, 11, 3), candidates) == frozenset(
      [ktype.DimValues({"b": (7, 11), "c": (3,)})]
  )
  # ... but not if the shape value in question is 1.
  assert spec.match((7, 1, 1), candidates) == candidates


def test_match_optional_dim():
  spec = shape_spec_parser.parse("a? b c? b b")
  assert spec.match((1, 1, 1), _empty) == {ktype.DimValues({"b": (1,)})}
  assert spec.match((2, 1, 1, 1), _empty) == {
      ktype.DimValues({"a": (2,), "b": (1,)})
  }
  assert spec.match((1, 2, 1, 1), _empty) == {
      ktype.DimValues({"c": (2,), "b": (1,)})
  }
  assert spec.match((3, 1, 2, 1, 1), _empty) == {
      ktype.DimValues({"a": (3,), "c": (2,), "b": (1,)})
  }


def test_infer_right_shape():
  candidates = frozenset([ktype.DimValues({"a": (7,)})])
  spec = shape_spec_parser.parse("a*b")
  m = spec.match((21,), candidates)
  assert m == {ktype.DimValues({"a": (7,), "b": (3,)})}

  spec = shape_spec_parser.parse("a+b")
  m = spec.match((12,), candidates)
  assert m == {ktype.DimValues({"a": (7,), "b": (5,)})}

  spec = shape_spec_parser.parse("a-b")
  m = spec.match((5,), candidates)
  assert m == {ktype.DimValues({"a": (7,), "b": (2,)})}

  spec = shape_spec_parser.parse("a//b")
  m = spec.match((1,), candidates)
  assert m == {ktype.DimValues({"a": (7,), "b": (7,)})}

  spec = shape_spec_parser.parse("a/b")
  m = spec.match((7,), candidates)
  assert m == {ktype.DimValues({"a": (7,), "b": (1,)})}


def test_infer_left_shape():
  candidates = frozenset([ktype.DimValues({"b": (7,)})])
  spec = shape_spec_parser.parse("a*b")
  m = spec.match((21,), candidates)
  assert m == {ktype.DimValues({"a": (3,), "b": (7,)})}

  spec = shape_spec_parser.parse("a+b")
  m = spec.match((12,), candidates)
  assert m == {ktype.DimValues({"a": (5,), "b": (7,)})}

  spec = shape_spec_parser.parse("a-b")
  m = spec.match((5,), candidates)
  assert m == {ktype.DimValues({"a": (12,), "b": (7,)})}

  spec = shape_spec_parser.parse("a//b")
  m = spec.match((2,), candidates)
  assert m == {ktype.DimValues({"a": (14,), "b": (7,)})}

  spec = shape_spec_parser.parse("a/b")
  m = spec.match((3,), candidates)
  assert m == {ktype.DimValues({"a": (21,), "b": (7,)})}
