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

import jax
import jax.numpy as jnp
from kauldron.ktyping import array_type_meta as atm
from kauldron.ktyping import array_types as art
from kauldron.ktyping import dtypes
import numpy as np
import pytest

# TODO(klausg): test tf and torch types


def test_array_type_creation():
  t = art.Float32["1 2 3"]
  assert t.shape_spec == "1 2 3"
  np_array = np.zeros((1, 2, 3), dtype=np.float32)
  jnp_array = jnp.zeros((1, 2, 3), dtype=np.float32)

  assert t.array_types_match(np_array)
  assert t.array_types_match(jnp_array)
  assert t.dtype_matches(np_array)
  assert t.dtype_matches(jnp_array)
  assert t.shape_matches(np_array)
  assert t.shape_matches(jnp_array)

  assert isinstance(np_array, t)
  assert isinstance(jnp_array, t)

  assert not t.dtype_matches(np.zeros((1, 2, 3), dtype=np.int32))
  assert not isinstance(np.zeros((1, 2, 3), dtype=np.int32), t)
  assert not isinstance(np.zeros((2, 3), dtype=np.float32), t)

  assert isinstance(np_array, art.Float)


def test_dtype_matches():
  assert art.Array.dtype_matches(np.empty((), dtype=np.bool_))
  assert art.Bool.dtype_matches(np.empty((), dtype=np.bool_))

  for dtype in (np.float16, np.float32, np.float64, jnp.bfloat16):
    assert art.Array.dtype_matches(np.empty((), dtype=dtype))
    assert art.Num.dtype_matches(np.empty((), dtype=dtype))
    assert art.Float.dtype_matches(np.empty((), dtype=dtype))

  assert art.Float16.dtype_matches(np.empty((), dtype=np.float16))
  assert art.Float32.dtype_matches(np.empty((), dtype=np.float32))
  assert art.Float64.dtype_matches(np.empty((), dtype=np.float64))
  assert art.BFloat16.dtype_matches(np.empty((), dtype=jnp.bfloat16))

  sint_types = (np.int8, np.int16, np.int32, np.int64)
  uint_types = (np.uint8, np.uint16, np.uint32, np.uint64)

  for dtype in sint_types:
    assert art.Array.dtype_matches(np.empty((), dtype=dtype))
    assert art.Num.dtype_matches(np.empty((), dtype=dtype))
    assert art.Int.dtype_matches(np.empty((), dtype=dtype))
    assert art.SInt.dtype_matches(np.empty((), dtype=dtype))

  assert art.Int8.dtype_matches(np.empty((), dtype=np.int8))
  assert art.Int16.dtype_matches(np.empty((), dtype=np.int16))
  assert art.Int32.dtype_matches(np.empty((), dtype=np.int32))
  assert art.Int64.dtype_matches(np.empty((), dtype=np.int64))

  for dtype in uint_types:
    assert art.Array.dtype_matches(np.empty((), dtype=dtype))
    assert art.Num.dtype_matches(np.empty((), dtype=dtype))
    assert art.Int.dtype_matches(np.empty((), dtype=dtype))
    assert art.UInt.dtype_matches(np.empty((), dtype=dtype))

  assert art.UInt8.dtype_matches(np.empty((), dtype=np.uint8))
  assert art.UInt16.dtype_matches(np.empty((), dtype=np.uint16))
  assert art.UInt32.dtype_matches(np.empty((), dtype=np.uint32))
  assert art.UInt64.dtype_matches(np.empty((), dtype=np.uint64))

  assert art.Array.dtype_matches(np.empty((), dtype=np.complex64))
  assert art.Num.dtype_matches(np.empty((), dtype=np.complex64))
  assert art.Complex.dtype_matches(np.empty((), dtype=np.complex64))
  assert art.Complex64.dtype_matches(np.empty((), dtype=np.complex64))


@pytest.mark.parametrize(
    "arr_type, dtype",
    (
        (art.Float, np.int32),
        (art.Float, np.bool_),
        (art.Float, np.complex64),
        (art.Float32, np.float64),
        (art.Float16, np.float32),
        (art.BFloat16, np.float16),
        (art.Int, np.float32),
        (art.Int32, np.int64),
        (art.Int32, np.uint32),
    ),
)
def test_dtype_mismatch(arr_type, dtype):
  assert not arr_type.dtype_matches(np.zeros((1, 2), dtype=dtype))


def test_is_array_type():
  assert atm.is_array_type(art.Array)  # type: ignore
  assert atm.is_array_type(art.Float32)  # type: ignore
  assert atm.is_array_type(art.Float32["1 2 3"])  # type: ignore
  assert atm.is_array_type(art.Float32["1 2 3"])  # type: ignore
  assert not atm.is_array_type(int)  # type: ignore
  assert not atm.is_array_type(np.float32)  # type: ignore


def test_scalar_types():
  assert isinstance(True, art.Scalar)
  assert isinstance(True, art.ScalarBool)
  assert isinstance(np.array(True, dtype=np.bool_), art.Scalar)
  assert isinstance(np.array(True, dtype=np.bool_), art.ScalarBool)

  assert isinstance(1, art.Scalar)
  assert isinstance(1, art.ScalarInt)
  assert isinstance(np.array(1, dtype=np.int32), art.Scalar)
  assert isinstance(np.array(1, dtype=np.int32), art.ScalarInt)

  assert isinstance(1.0, art.Scalar)
  assert isinstance(1.0, art.ScalarFloat)
  assert isinstance(np.array(1.0, dtype=np.float32), art.Scalar)
  assert isinstance(np.array(1.0, dtype=np.float32), art.ScalarFloat)

  assert isinstance(1j, art.Scalar)
  assert isinstance(1j, art.ScalarComplex)
  assert isinstance(np.array(1j, dtype=np.complex64), art.Scalar)
  assert isinstance(np.array(1j, dtype=np.complex64), art.ScalarComplex)

  # not a scalar if it has a shape
  assert not isinstance(np.zeros((2,)), art.Scalar)
  assert not isinstance(np.zeros((1,)), art.Scalar)


def test_long_form_array_types():

  assert art.Float32[np.ndarray, "1 2 3"].array_types == (np.ndarray,)
  assert art.Float32[(np.ndarray, jnp.ndarray), "1 2 3"].array_types == (
      np.ndarray,
      jnp.ndarray,
  )
  assert art.Array[np.ndarray, "a", dtypes.float32].dtype == dtypes.float32
  assert (
      art.Array[(np.ndarray, jnp.ndarray), "a", dtypes.float32].dtype
      == dtypes.float32
  )

  with pytest.raises(TypeError):
    _ = art.Float32[np.ndarray, "a", dtypes.float64]

  with pytest.raises(TypeError):
    _ = art.Scalar[np.ndarray, "a"]

  with pytest.raises(TypeError):
    _ = art.Float32["a", np.ndarray]

  with pytest.raises(TypeError):
    NpFloat32 = atm.ArrayTypeMeta(  # pylint: disable=invalid-name
        "NpFloat32", array_types=(np.ndarray,), dtype=dtypes.float32
    )

    _ = NpFloat32[jnp.ndarray, "*b"]


def test_array_type_or_dtypes():
  FloatInt32 = (art.Float32 | art.Int32)["*b"]  # pylint: disable=invalid-name
  assert isinstance(FloatInt32, atm.ArrayTypeMeta)  # and not a UnionType
  assert FloatInt32.array_types == art.Float32.array_types
  assert FloatInt32.dtype == dtypes.float32 | dtypes.int32
  assert FloatInt32.shape_spec == "*b"

  assert FloatInt32.dtype_matches(np.zeros((1, 2), dtype=np.float32))
  assert FloatInt32.dtype_matches(jnp.zeros((1, 2), dtype=np.int32))
  assert not FloatInt32.dtype_matches(np.zeros((1, 2), dtype=np.bool_))


def test_array_type_or_other():
  NpArray = atm.ArrayTypeMeta("NpFloat32", array_types=(atm.NpArray,))  # pylint: disable=invalid-name
  NpFloat32abc = (NpArray | art.Float32)["a b c"]  # pylint: disable=invalid-name
  assert isinstance(NpFloat32abc, atm.ArrayTypeMeta)  # and not a UnionType
  assert NpFloat32abc.array_types == (atm.NpArray,)
  assert NpFloat32abc.dtype == dtypes.float32
  assert NpFloat32abc.shape_spec == "a b c"


def test_array_types_not_callable():
  with pytest.raises(RuntimeError, match="cannot be instantiated"):
    art.Float32("a b")


def test_check_array_type_naming():
  # Test that the name of the ArrayTypeMeta class matches its symbol
  # This is to catch typos like:
  #   Complex128 = atm.ArrayTypeMeta("Complex64", dtype=dtypes.float32)
  for name, val in art.__dict__.items():
    if isinstance(val, atm.ArrayTypeMeta):
      assert name == val.__name__


def test_shape():
  assert isinstance([1, 2, 3], art.Shape)
  assert isinstance((1, 2, 3), art.Shape)
  assert not isinstance([0.2, None, 3], art.Shape)


def test_shape_call_raises():
  with pytest.raises(RuntimeError, match="cannot be instantiated"):
    art.Shape("b n")


def test_xarray_accepts_any_dtype():
  for dtype in (np.float32, np.int32, np.bool_, np.complex64, np.uint8):
    assert art.XArray.dtype_matches(np.empty((), dtype=dtype))
    assert isinstance(np.zeros((2, 3), dtype=dtype), art.XArray)


def test_prng_key_array_dtype():
  key = jax.random.key(0)
  assert art.PRNGKeyArray.dtype_matches(key)
  assert art.PRNGKey.dtype_matches(key)

  keys = jax.random.split(key, 4)
  assert art.PRNGKeyArray.dtype_matches(keys)
  assert isinstance(keys, art.PRNGKeyArray)

  assert not art.Float32.dtype_matches(key)
  assert not art.Int.dtype_matches(key)
