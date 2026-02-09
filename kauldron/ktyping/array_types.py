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

"""Common array type definitions."""

from kauldron.ktyping import array_type_meta as atm
from kauldron.ktyping import dtypes
from kauldron.ktyping import internal_typing

MISSING = internal_typing.MISSING

# MARK: jnp / np ArrayTypes
# These (default) array types accept both jax and numpy arrays.

# Generic array type.
Array = atm.ArrayTypeMeta("Array")

# Any numerical type: int, float, complex, etc. but NOT bool.
# see also https://numpy.org/doc/2.1/reference/arrays.scalars.html#scalars
Num = atm.ArrayTypeMeta("Num", dtype=dtypes.number)

# Boolean Array
Bool = atm.ArrayTypeMeta("Bool", dtype=dtypes.bool_)

# Integer types (signed and unsigned below. NOT including bool.)
Int = atm.ArrayTypeMeta("Int", dtype=dtypes.integer)
# Integer or Bool
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
Complex128 = atm.ArrayTypeMeta("Complex128", dtype=dtypes.complex128)


# RNG keys
# Support both uint32 and the newer jax.dtypes.prng_key dtypes.
# See: https://docs.jax.dev/en/latest/jep/9263-typed-keys.html
PRNGKey = atm.ArrayTypeMeta(
    "PRNGKey", dtype=dtypes.prng_key | dtypes.uint32, shape_spec="2"
)

# Only support new jax.dtypes.prng_key dtype for the array of prgn keys.
PRNGKeyArray = atm.ArrayTypeMeta("PRNGKeyArray", dtype=dtypes.prng_key)


# MARK: tf ArrayTypes


def _create_tf_type(name, dtype=MISSING):
  return atm.ArrayTypeMeta(name, dtype=dtype, array_types=(atm.TfArray,))


# Generic array type.
TfArray = _create_tf_type("TfArray")

# Any numerical type: int, float, complex, etc. but NOT bool.
# see also https://numpy.org/doc/2.1/reference/arrays.scalars.html#scalars
TfNum = _create_tf_type("TfNum", dtypes.number)

# Boolean Array
TfBool = _create_tf_type("TfBool", dtypes.bool_)

# Integer types (signed and unsigned below. NOT including bool.)
TfInt = _create_tf_type("TfInt", dtypes.integer)
# Integer or Bool
TfIntOrBool = _create_tf_type("TfIntOrBool", dtypes.bool_ | dtypes.integer)

# signed
TfSInt = _create_tf_type("TfSInt", dtypes.signedinteger)
TfInt8 = _create_tf_type("TfInt8", dtypes.int8)
TfInt16 = _create_tf_type("TfInt16", dtypes.int16)
TfInt32 = _create_tf_type("TfInt32", dtypes.int32)
TfInt64 = _create_tf_type("TfInt64", dtypes.int64)
# unsigned
TfUInt = _create_tf_type("TfUInt", dtypes.unsignedinteger)
TfUInt8 = _create_tf_type("TfUInt8", dtypes.uint8)
TfUInt16 = _create_tf_type("TfUInt16", dtypes.uint16)
TfUInt32 = _create_tf_type("TfUInt32", dtypes.uint32)
TfUInt64 = _create_tf_type("TfUInt64", dtypes.uint64)

# Float types
TfFloat = _create_tf_type("TfFloat", dtypes.floating)
TfFloat16 = _create_tf_type("TfFloat16", dtypes.float16)
TfFloat32 = _create_tf_type("TfFloat32", dtypes.float32)
TfFloat64 = _create_tf_type("TfFloat64", dtypes.float64)

# Complex types
TfComplex = _create_tf_type("TfComplex", dtypes.complex64 | dtypes.complex128)
TfComplex64 = _create_tf_type("TfComplex64", dtypes.complex64)
TfComplex128 = _create_tf_type("TfComplex128", dtypes.complex128)


# MARK: x Arraytypes
# Array types that accept nump, jax, tensorflow and torch arrays.
def _create_x_type(name, dtype=MISSING):
  return atm.ArrayTypeMeta(
      name,
      dtype=dtype,
      array_types=(atm.NpArray, atm.JaxArray, atm.TfArray, atm.TorchArray),
  )


# Generic array type.
XArray = _create_x_type("XArray", None)

# Any numerical type: int, float, complex, etc. but NOT bool.
# see also https://numpy.org/doc/2.1/reference/arrays.scalars.html#scalars
XNum = _create_x_type("XNum", dtypes.number)

# Boolean Array
XBool = _create_x_type("XBool", dtypes.bool_)

# Integer types (signed and unsigned below. NOT including bool.)
XInt = _create_x_type("XInt", dtypes.integer)
# Integer or Bool
XIntOrBool = _create_x_type("XIntOrBool", dtypes.bool_ | dtypes.integer)

# signed
XSInt = _create_x_type("XSInt", dtypes.signedinteger)
XInt8 = _create_x_type("XInt8", dtypes.int8)
XInt16 = _create_x_type("XInt16", dtypes.int16)
XInt32 = _create_x_type("XInt32", dtypes.int32)
XInt64 = _create_x_type("XInt64", dtypes.int64)
# unsigned
XUInt = _create_x_type("XUInt", dtypes.unsignedinteger)
XUInt8 = _create_x_type("XUInt8", dtypes.uint8)
XUInt16 = _create_x_type("XUInt16", dtypes.uint16)
XUInt32 = _create_x_type("XUInt32", dtypes.uint32)
XUInt64 = _create_x_type("XUInt64", dtypes.uint64)

# Float types
XFloat = _create_x_type("XFloat", dtypes.floating)
XFloat16 = _create_x_type("XFloat16", dtypes.float16)
XFloat32 = _create_x_type("XFloat32", dtypes.float32)
XFloat64 = _create_x_type("XFloat64", dtypes.float64)


# Complex types
XComplex = _create_x_type("XComplex", dtypes.complex64 | dtypes.complex128)
XComplex64 = _create_x_type("XComplex64", dtypes.complex64)
XComplex128 = _create_x_type("XComplex128", dtypes.complex128)


# MARK: Scalar types
# These are either python builtin scalars or np/jnp/tf/torch scalars
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


# MARK: Shape types
Shape = atm.ShapeMeta("Shape")
