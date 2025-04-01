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

"""Lazy dtypes."""

from __future__ import annotations

import abc
import dataclasses
import sys
from typing import Any

from etils.enp import lazy  # pylint: disable=g-importing-member
import numpy as np


__all__ = (
    "DType",
    "pybool",
    "pyint",
    "pyfloat",
    "pycomplex",
    "generic",
    "number",
    "bool_",
    "floating",
    "float16",
    "float32",
    "float64",
    "float128",
    "integer",
    "signedinteger",
    "int8",
    "int16",
    "int32",
    "int64",
    "unsignedinteger",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
    "float0",
    "bfloat16",
    "prng_key",
    "bool_like",
    "int_like",
    "float_like",
    "complex_like",
    "scalar_like",
)


class DType(abc.ABC):

  @abc.abstractmethod
  def matches(self, obj: Any) -> bool:
    """Returns True if the object (array or scalar) matches this dtype."""

  def __or__(self, other: Any) -> CompoundDType | NotImplemented:
    if not isinstance(other, DType):
      return NotImplemented
    return CompoundDType(subtypes=(self, other))


@dataclasses.dataclass(frozen=True)
class BuiltinDType(DType):
  """Dtype that matches a builtin scalar type."""

  types_: type[Any] | tuple[type[Any], ...]

  def matches(self, obj: Any) -> bool:
    return isinstance(obj, self.types_)


@dataclasses.dataclass(frozen=True)
class NpDType(DType):
  """Numpy dtype."""

  dtype: Any

  def matches(self, obj: Any) -> bool:
    if not hasattr(obj, "dtype"):
      return False
    obj_dtype = lazy.as_np_dtype(obj.dtype)  # convert tf and torch dtypes
    if "jax" in sys.modules:
      # use jax.dtypes.issubdtype if jax is available
      # this is needed so that e.g. bfloat16 will be matched by np.number
      import jax  # pylint: disable=g-import-not-at-top

      return jax.dtypes.issubdtype(obj_dtype, self.dtype)
    else:
      return np.issubdtype(obj_dtype, self.dtype)


@dataclasses.dataclass(frozen=True)
class JaxDType(DType):
  """Dtype that imports the parent module (e.g. jax.dtypes) lazily."""

  dtype: str

  def matches(self, obj: Any) -> bool:
    if not hasattr(obj, "dtype"):
      return False
    obj_dtype = lazy.as_jax_dtype(obj.dtype)  # convert tf and torch dtypes
    try:
      import jax  # pylint: disable=g-import-not-at-top

      target_dtype = getattr(jax.dtypes, self.dtype)
      return jax.dtypes.issubdtype(obj_dtype, target_dtype)
    except ImportError:
      return False


@dataclasses.dataclass(frozen=True)
class CompoundDType(DType):
  """Dtype that matches any of the subtypes."""

  subtypes: tuple[DType, ...]

  def matches(self, obj: Any) -> bool:
    return any(sub.matches(obj) for sub in self.subtypes)

  def __post_init__(self):
    dtypes = []
    # flatten the subtypes
    for st in self.subtypes:
      if isinstance(st, CompoundDType):
        dtypes.extend(st.subtypes)
      else:
        dtypes.append(st)
    object.__setattr__(self, "subtypes", tuple(dtypes))


# builtin scalar types
pybool = BuiltinDType(bool)
pyint = BuiltinDType(int)
pyfloat = BuiltinDType(float)
pycomplex = BuiltinDType(complex)

# numpy dtypes
generic = NpDType(np.generic)
number = NpDType(np.number)
bool_ = NpDType(np.bool_)
floating = NpDType(np.floating)
float16 = NpDType(np.float16)
float32 = NpDType(np.float32)
float64 = NpDType(np.float64)
float128 = NpDType(np.float128)
integer = NpDType(np.integer)
signedinteger = NpDType(np.signedinteger)
int8 = NpDType(np.int8)
int16 = NpDType(np.int16)
int32 = NpDType(np.int32)
int64 = NpDType(np.int64)
unsignedinteger = NpDType(np.unsignedinteger)
uint8 = NpDType(np.uint8)
uint16 = NpDType(np.uint16)
uint32 = NpDType(np.uint32)
uint64 = NpDType(np.uint64)
complex64 = NpDType(np.complex64)
complex128 = NpDType(np.complex128)

# jax dtypes
float0 = JaxDType("float0")
bfloat16 = JaxDType("bfloat16")
prng_key = JaxDType("prng_key")


# scalar dtypes
bool_like = pybool | bool_
int_like = pyint | integer
float_like = pyfloat | floating | bfloat16 | float0
complex_like = pycomplex | complex64 | complex128

scalar_like = bool_like | int_like | float_like | complex_like
