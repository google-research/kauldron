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

from kauldron.ktyping import shape_spec_parser
# pylint: disable=g-importing-member
from kauldron.ktyping.shape_spec import (  # pylint: disable=g-multiple-import
    AnonDims,
    BinaryOpDim,
    ChoiceDim,
    FunctionDim,
    IntDim,
    NAME_2_FUNC,
    NamedDims,
    NegatedDim,
    ShapeSpec,
    SYMBOL_2_OPERATOR,
)
import pytest
# pylint: enable=g-importing-member


SHAPE_SPEC_EXAMPLES = [
    ("", ShapeSpec()),
    (
        "*batch",
        ShapeSpec(NamedDims("batch", length=None)),
    ),
    ("...", ShapeSpec(AnonDims(length=None))),
    (
        "*#b",
        ShapeSpec(NamedDims("b", length=None, broadcastable=True)),
    ),
    (
        "_a #b +third 42",
        ShapeSpec(
            AnonDims("a"),
            NamedDims("b", broadcastable=True),
            NamedDims("third", length=(1, None)),
            IntDim(42),
        ),
    ),
    (
        "*b h//2 min(h,w) c+1",
        ShapeSpec(
            NamedDims("b", length=None),
            BinaryOpDim(
                op=SYMBOL_2_OPERATOR["//"],
                left=NamedDims("h"),
                right=IntDim(2),
            ),
            FunctionDim(
                "min",
                NAME_2_FUNC["min"],
                [NamedDims("h"), NamedDims("w")],
            ),
            BinaryOpDim(
                op=SYMBOL_2_OPERATOR["+"],
                left=NamedDims("c"),
                right=IntDim(1),
            ),
        ),
    ),
    (
        "2*a+b**-c",
        ShapeSpec(
            BinaryOpDim(
                SYMBOL_2_OPERATOR["+"],
                left=BinaryOpDim(
                    SYMBOL_2_OPERATOR["*"],
                    left=IntDim(2),
                    right=NamedDims("a"),
                ),
                right=BinaryOpDim(
                    SYMBOL_2_OPERATOR["**"],
                    left=NamedDims("b"),
                    right=NegatedDim(NamedDims("c")),
                ),
            )
        ),
    ),
    (
        "+#a b|c",
        ShapeSpec(
            NamedDims("a", length=(1, None), broadcastable=True),
            ChoiceDim(left=NamedDims("b"), right=NamedDims("c")),
        ),
    ),
]


@pytest.mark.parametrize("spec_str,expected_spec", SHAPE_SPEC_EXAMPLES)
def test_shape_parser(spec_str, expected_spec):
  parsed_spec = shape_spec_parser.parse(spec_str)
  assert parsed_spec == expected_spec
  assert repr(expected_spec) == spec_str


@pytest.mark.parametrize(
    "spec_str",
    [
        "a + b",
        "a b + c",
        "a b c + d",
        "a b c d + e",
        "a - b",
        "a b - c",
        "a b c - d",
        "a b c d - e",
        "a * b",
        "a b * c",
        "a b c * d",
        "a b c d * e",
        "a / b",
        "a b / c",
        "a b c / d",
        "a b c d / e",
        "a % b",
        "a b % c",
        "a b c % d",
        "a b c d % e",
        "a // b",
        "a b // c",
        "a b c // d",
        "a b c d // e",
        "a ** b",
        "a b ** c",
        "a b c ** d",
        "a b c d ** e",
        "(a + b)",
        "a (b + 1)",
        "*batch foo * bar",
        "b (foo + 1)*bar",
        "b // 2",
        "sum(a, b)",
        "a sum(b, c)",
        "(a b)",
        "(a b c)",
        "*b (foo bar)",
        "prefix *b (foo bar) #c",
        "a b c d e f (g h i+i)",
        "a b (c+1 d)",
        "prod(a, b)",
        "a - b",
    ],
)
def test_ws_syntax_error(spec_str: str):
  with pytest.raises(shape_spec_parser.WhiteSpaceError):
    shape_spec_parser.parse(spec_str)


@pytest.mark.parametrize(
    "spec_str",
    [
        "a,b",
        "a, b, c",
        "a b, c",
        "*b, c",
        "*b , c",
        "a ,b",
        "(a+1), b",
        "a (b+c), d",
        "sum(*b,h), min(w,4) 2*c+h",
        "a b c d e f (g+i), j k l m",
    ],
)
def test_comma_syntax_error(spec_str: str):
  with pytest.raises(shape_spec_parser.InvalidCommaError):
    shape_spec_parser.parse(spec_str)


@pytest.mark.parametrize(
    "spec_str",
    [
        "_4",
        "*b c _2 d e",
        "a b c d e _192",
    ],
)
def test_fixed_anon_dim_error(spec_str: str):
  with pytest.raises(
      shape_spec_parser.AnonymousFixedDimError, match="at position"
  ):
    shape_spec_parser.parse(spec_str)


@pytest.mark.parametrize(
    "spec_str",
    [
        "_#a",
        "#_a",
        "*b c _#a d e",
        "*b c #_a d e",
        "a b c d e #_f",
        "a b c d e _#f",
    ],
)
def test_fixed_anon_broadcastable_dim_error(spec_str: str):
  with pytest.raises(shape_spec_parser.AnonymousBroadcastableDimError):
    shape_spec_parser.parse(spec_str)


@pytest.mark.parametrize(
    "spec_str",
    [
        "a* b",
        "a b* c",
        "a# b",
        "a b# c",
        "a*# b",
        "a#* b",
        "#a* b",
        "*a# b",
        "a b*# c",
        "a b#* c",
        "a #b* c",
        "a *b# c",
        "sum(a*, b) c",
        "a sum(b*, c) d",
    ],
)
def test_post_star_error(spec_str: str):
  with pytest.raises(shape_spec_parser.PostModifierError):
    shape_spec_parser.parse(spec_str)


def test_unknon_syntax_error():
  with pytest.raises(
      shape_spec_parser.ShapeSpecSyntaxError, match="expected"
  ) as exc_info:
    shape_spec_parser.parse("$a b c d")
  assert exc_info.type is shape_spec_parser.ShapeSpecSyntaxError


# @pytest.mark.skip("no longer a syntax error")
# @pytest.mark.parametrize(
#     "spec_str",
#     [
#         "*b+2",
#         "1+*b",
#         "-*c",
#         "a *b-1",
#         "a b *c//2",
#         "2***b",
#         "a b c d e *f+1 g h i",
#     ],
# )
# def test_multi_dim_op_error(spec_str: str):
#   with pytest.raises(shape_spec_parser.MultiDimOperationError):
#     shape_spec_parser.parse(spec_str)
#   pass
