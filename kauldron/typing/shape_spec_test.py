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

# pylint: disable=g-importing-member
from kauldron.typing import Float, Shape, typechecked  # pylint: disable=g-multiple-import
from kauldron.typing.shape_parser import (  # pylint: disable=g-multiple-import
    BinaryOpDim,
    FunctionDim,
    IntDim,
    NAME_2_FUNC,
    NegDim,
    SYMBOL_2_OPERATOR,
    ShapeSpec,
    SingleDim,
    VariadicDim,
)
from kauldron.typing.shape_spec import Dim, parse_shape_spec  # pylint: disable=g-multiple-import
from kauldron.typing.utils import Memo
import numpy as np
import pytest


SHAPE_SPEC_EXAMPLES = [
    ("", ShapeSpec()),
    ("*batch", ShapeSpec(VariadicDim("batch"))),
    ("...", ShapeSpec(VariadicDim(anonymous=True))),
    ("*#b", ShapeSpec(VariadicDim("b", broadcastable=True))),
    (
        "_a #b third 42",
        ShapeSpec(
            SingleDim("a", anonymous=True),
            SingleDim("b", broadcastable=True),
            SingleDim("third"),
            IntDim(42),
        ),
    ),
    (
        "*b h//2 min(h,w) c+1",
        ShapeSpec(
            VariadicDim("b"),
            BinaryOpDim(
                op=SYMBOL_2_OPERATOR["//"], left=SingleDim("h"), right=IntDim(2)
            ),
            FunctionDim(
                "min", NAME_2_FUNC["min"], [SingleDim("h"), SingleDim("w")]
            ),
            BinaryOpDim(
                op=SYMBOL_2_OPERATOR["+"], left=SingleDim("c"), right=IntDim(1)
            ),
        ),
    ),
    (
        "2*a+b**-c",
        ShapeSpec(
            BinaryOpDim(
                SYMBOL_2_OPERATOR["+"],
                left=BinaryOpDim(
                    SYMBOL_2_OPERATOR["*"], left=IntDim(2), right=SingleDim("a")
                ),
                right=BinaryOpDim(
                    SYMBOL_2_OPERATOR["**"],
                    left=SingleDim("b"),
                    right=NegDim(SingleDim("c")),
                ),
            )
        ),
    ),
]


@pytest.mark.parametrize("spec_str,expected_spec", SHAPE_SPEC_EXAMPLES)
def test_shape_parser(spec_str, expected_spec):
  parsed_spec = parse_shape_spec(spec_str)
  assert parsed_spec == expected_spec
  assert repr(expected_spec) == spec_str


def test_shape_eval():

  @typechecked
  def _foo1():
    return Shape("2 5*6+2 8//2 5-4+1 12/3*2")

  assert _foo1() == (2, 32, 4, 2, 8)

  @typechecked
  def _foo(_: Float["*b h w c"]):
    return Shape("sum(*b,h) h//2 min(w,4) c+1")

  assert _foo(np.zeros((1, 2, 3, 4))) == (3, 1, 3, 5)


def test_dim():
  @typechecked
  def _foo(_: Float["*b h w c"]):
    return [Dim(d) for d in ["*b", "h", "w", "c"]]

  assert _foo(np.zeros((1, 2, 3, 4))) == [1, 2, 3, 4]


def test_shape_eval_with_batch_dim():
  memo = Memo({"n": 16}, {"batch": (3, 2)})
  parsed_shape = parse_shape_spec("*batch n")
  assert parsed_shape.evaluate(memo) == (3, 2, 16)


def test_shape_eval_error_outside_typechecked():
  with pytest.raises(AssertionError):
    Shape("1 2 3")

  def _foo():
    return Dim("1")

  with pytest.raises(AssertionError):
    _foo()
