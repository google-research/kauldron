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

"""Common array type definitions."""

from typing import Any, TypeGuard
from kauldron.ktyping import array_type_meta as atm
from kauldron.ktyping import dtypes

__all__ = (
    "Array",
    "Num",
    "Bool",
    "Int",
    "IntOrBool",
    "SInt",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "BFloat16",
    "Complex",
    "Complex64",
    "Complex128",
    "Scalar",
    "ScalarBool",
    "ScalarInt",
    "ScalarFloat",
    "ScalarComplex",
    "PRNGKey",
    "PRNGKeyArray",
    "is_array_type",
)


def is_array_type(origin_type: Any) -> TypeGuard[atm.ArrayTypeMeta]:
  return isinstance(origin_type, atm.ArrayTypeMeta)


# Generic array type.
Array = atm.ArrayTypeMeta("Array")

# Any numerical type
Num = atm.ArrayTypeMeta("Num", dtype=dtypes.number)

# Boolean Array
Bool = atm.ArrayTypeMeta("Bool", dtype=dtypes.bool_)

# Integer types
Int = atm.ArrayTypeMeta("Int32", dtype=dtypes.integer)
IntOrBool = atm.ArrayTypeMeta("IntOrBool", dtype=dtypes.bool_ | dtypes.integer)

# signed
SInt = atm.ArrayTypeMeta("SInt", dtype=dtypes.signedinteger)
Int8 = atm.ArrayTypeMeta("Int8", dtype=dtypes.int8)
Int16 = atm.ArrayTypeMeta("Int16", dtype=dtypes.int16)
Int32 = atm.ArrayTypeMeta("Int32", dtype=dtypes.int32)
Int64 = atm.ArrayTypeMeta("Int64", dtype=dtypes.int64)
# unsigned
UInt = atm.ArrayTypeMeta("UInt", dtype=dtypes.unsignedinteger)
UInt8 = atm.ArrayTypeMeta("UInt8", dtype=dtypes.uint8)
UInt16 = atm.ArrayTypeMeta("UInt16", dtype=dtypes.uint16)
UInt32 = atm.ArrayTypeMeta("UInt32", dtype=dtypes.uint32)
UInt64 = atm.ArrayTypeMeta("UInt64", dtype=dtypes.uint64)

# Float types
Float = atm.ArrayTypeMeta("Float", dtype=dtypes.floating)
Float16 = atm.ArrayTypeMeta("Float16", dtype=dtypes.float16)
Float32 = atm.ArrayTypeMeta("Float32", dtype=dtypes.float32)
Float64 = atm.ArrayTypeMeta("Float64", dtype=dtypes.float64)
BFloat16 = atm.ArrayTypeMeta("BFloat16", dtype=dtypes.bfloat16)

# Complex types
Complex = atm.ArrayTypeMeta(
    "Complex", dtype=dtypes.complex64 | dtypes.complex128
)
Complex64 = atm.ArrayTypeMeta("Complex64", dtype=dtypes.complex64)
Complex128 = atm.ArrayTypeMeta("Complex64", dtype=dtypes.complex128)


# TODO(klausg): text types?


# Scalar types
Scalar = atm.ArrayTypeMeta(
    "Scalar",
    array_types=atm.ScalarLike,
    dtype=dtypes.scalar_like,
    shape_spec="",
)
ScalarBool = atm.ArrayTypeMeta(
    "ScalarBool",
    array_types=atm.ScalarLike,
    dtype=dtypes.bool_like,
    shape_spec="",
)
ScalarInt = atm.ArrayTypeMeta(
    "ScalarInt",
    array_types=atm.ScalarLike,
    dtype=dtypes.int_like,
    shape_spec="",
)
ScalarFloat = atm.ArrayTypeMeta(
    "ScalarFloat",
    array_types=atm.ScalarLike,
    dtype=dtypes.float_like,
    shape_spec="",
)
ScalarComplex = atm.ArrayTypeMeta(
    "ScalarComplex",
    array_types=atm.ScalarLike,
    dtype=dtypes.complex_like,
    shape_spec="",
)

# RNG keys
# Support both uint32 and the newer jax.dtypes.prng_key dtypes.
# See: https://docs.jax.dev/en/latest/jep/9263-typed-keys.html
PRNGKey = atm.ArrayTypeMeta(
    "PRNGKey", dtype=dtypes.prng_key | dtypes.uint32, shape_spec="2"
)  # TODO(klausg): support kd.random.PRNGKey?

# Only support new jax.dtypes.prng_key dtype for the array of prgn keys.
PRNGKeyArray = atm.ArrayTypeMeta("PRNGKeyArray", dtype=dtypes.prng_key)
