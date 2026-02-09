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

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from kauldron.typing.array_types import Array, Bool, Float, Float32, Int, UInt32, UInt8  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import pytest


class ShapeExample(NamedTuple):
  shape_spec: str
  pos_examples: list[tuple[int, ...]]
  neg_examples: list[tuple[int, ...]]


SHAPE_EXAMPLES = [
    ShapeExample(
        shape_spec="1",
        pos_examples=[(1,)],
        neg_examples=[(2,), (1, 1), ()],
    ),
    ShapeExample(
        shape_spec="x",
        pos_examples=[(1,), (18,)],
        neg_examples=[(1, 1), ()],
    ),
    ShapeExample(
        shape_spec="h w 3",
        pos_examples=[(1, 2, 3), (18, 18, 3)],
        neg_examples=[(3, 3), (32, 32, 4), (1, 2, 3, 4)],
    ),
    ShapeExample(
        shape_spec="h h/2 h+1",
        pos_examples=[(2, 1, 3), (18, 9, 19)],
        neg_examples=[(5, 2, 6), (2, 2), (2, 2, 2)],
    ),
    ShapeExample(
        shape_spec="*b x 2",
        pos_examples=[(1, 2), (1, 2, 2), (1, 4, 1, 4, 1, 2)],
        neg_examples=[(2,), (1, 1, 3)],
    ),
    ShapeExample(
        shape_spec="... x 2",
        pos_examples=[(1, 2), (1, 2, 2), (1, 4, 1, 4, 9, 2)],
        neg_examples=[(2,), (1, 1, 3)],
    ),
    ShapeExample(
        shape_spec="#b b",
        pos_examples=[(2, 2), (8, 8), (1, 8), (1, 1)],
        neg_examples=[(2, 4), (8, 1), (1,)],
    ),
    ShapeExample(
        shape_spec="2 _b _b",
        pos_examples=[(2, 3, 4)],
        neg_examples=[(2, 2)],
    ),
]

_FLAT_POSITIVE_SHAPE_EXAMPLES = [
    (ex.shape_spec, p) for ex in SHAPE_EXAMPLES for p in ex.pos_examples  # pylint: disable=g-complex-comprehension
]


@pytest.mark.parametrize("spec,shape", _FLAT_POSITIVE_SHAPE_EXAMPLES)
def test_positive_isinstance_examples(spec, shape):
  nx = np.zeros(shape)
  jx = np.zeros(shape)
  assert isinstance(nx, Float[spec])
  assert isinstance(jx, Float[spec])


_FLAT_NEGATIVE_SHAPE_EXAMPLES = [
    (ex.shape_spec, n) for ex in SHAPE_EXAMPLES for n in ex.neg_examples  # pylint: disable=g-complex-comprehension
]


@pytest.mark.parametrize("spec,shape", _FLAT_NEGATIVE_SHAPE_EXAMPLES)
def test_negative_isinstance_examples(spec, shape):
  nx = np.zeros(shape)
  jx = np.zeros(shape)
  assert not isinstance(nx, Float[spec])
  assert not isinstance(jx, Float[spec])


VALID_NP_DTYPES = {
    Float: [np.float16, np.float32, np.float64],
    Float32: [np.float32],
    Int: [np.int8, np.int16, np.int32, np.int64],
    Bool: [np.bool_],
    UInt8: [np.uint8],
    UInt32: [np.uint32],
}

_FLAT_NP_DTYPES = [
    (arr_type, dtype)  # pylint: disable=g-complex-comprehension
    for arr_type, dtypes in VALID_NP_DTYPES.items()
    for dtype in dtypes
]


@pytest.mark.parametrize("SpecType,dtype", _FLAT_NP_DTYPES)
def test_numpy_dtypes(SpecType, dtype):  # pylint: disable=invalid-name
  nx = np.zeros((1,), dtype=dtype)
  assert isinstance(nx, SpecType["x"])


VALID_JNP_DTYPES = {
    Float: [jnp.float16, jnp.float32],
    Float32: [jnp.float32],
    Int: [jnp.int8, jnp.int16, jnp.int32],
    Bool: [jnp.bool_],
    UInt8: [jnp.uint8],
    UInt32: [jnp.uint32],
}
_FLAT_JNP_DTYPES = [
    (arr_type, dtype)  # pylint: disable=g-complex-comprehension
    for arr_type, dtypes in VALID_NP_DTYPES.items()
    for dtype in dtypes
]


@pytest.mark.parametrize("SpecType,dtype", _FLAT_JNP_DTYPES)
def test_jnp_dtypes(SpecType, dtype):  # pylint: disable=invalid-name
  jx = jnp.zeros((1,), dtype=dtype)
  assert isinstance(jx, SpecType["x"])


def test_plain_array_isinstance():
  a = np.zeros((1,), dtype=np.float32)
  assert isinstance(a, Array)
  assert isinstance(a, Float)
  assert isinstance(a, Float32)
  assert not isinstance(a, Bool)
  assert not isinstance(a, Int)
