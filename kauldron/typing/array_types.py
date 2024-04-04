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

"""Array types for annotations and isinstance checking."""

from __future__ import annotations

import functools
import operator
from typing import Any

import jaxtyping
import numpy as np
import tensorflow as tf


# Inspired by https://github.com/google/etils/tree/HEAD/etils/enp/array_types/typing.py
class ArrayAliasMeta(type):
  """Metaclass to dynamically create array aliases.

  Required to confuse pytype enough to ignore the non-standard Array["..."]
  syntax.
  """

  dtype: Any
  array_types: tuple[Any, ...]

  def __new__(
      mcs,
      name: str,
      jaxtyping_type: Any,
      array_types=(jaxtyping.Array, np.ndarray),
  ):
    return super().__new__(
        mcs, name, (), {"dtype": jaxtyping_type, "array_types": array_types}
    )

  def __init__(cls, name: str, dtype: Any, array_types: tuple[Any, ...] = ()):
    del name, dtype, array_types
    super().__init__(cls)

  def __instancecheck__(cls, inst: Any):
    return isinstance(inst, cls["..."])

  def __getitem__(cls, item):
    # We construct Shaped[arr_type1, item] | Shaped[arr_type2, item] rather than
    # Shaped[Union[array_types], item] because of this issue:
    # https://github.com/google/jaxtyping/issues/73
    subtypes = [cls.dtype[t, item] for t in cls.array_types]
    for st in subtypes:
      # Preserve the kauldron name for the Array type for later error messages.
      st._kd_repr = f"{cls.__name__}['{item}']"
    return functools.reduce(operator.or_, subtypes)


Array = ArrayAliasMeta("Array", jaxtyping.Shaped)
Bool = ArrayAliasMeta("Bool", jaxtyping.Bool)
Complex = ArrayAliasMeta("Complex", jaxtyping.Complex)
Complex64 = ArrayAliasMeta("Complex64", jaxtyping.Complex64)
Float = ArrayAliasMeta("Float", jaxtyping.Float)
Float32 = ArrayAliasMeta("Float32", jaxtyping.Float32)
Int = ArrayAliasMeta("Int", jaxtyping.Int)
UInt8 = ArrayAliasMeta("UInt8", jaxtyping.UInt8)
UInt32 = ArrayAliasMeta("UInt32", jaxtyping.UInt32)
Integer = ArrayAliasMeta("Integer", jaxtyping.Integer)
Num = ArrayAliasMeta("Num", jaxtyping.Num)

Scalar = Array[""]
ScalarFloat = Float[""]
ScalarInt = Integer[""]

TfArray = ArrayAliasMeta("TfArray", jaxtyping.Shaped, array_types=(tf.Tensor,))
TfFloat = ArrayAliasMeta("TfFloat", jaxtyping.Float, array_types=(tf.Tensor,))
TfFloat32 = ArrayAliasMeta(
    "TfFloat32", jaxtyping.Float32, array_types=(tf.Tensor,)
)
TfInt = ArrayAliasMeta("TfInt", jaxtyping.Int, array_types=(tf.Tensor,))
TfUInt8 = ArrayAliasMeta("TfUInt8", jaxtyping.UInt8, array_types=(tf.Tensor,))


_tf_np_jnp = (tf.Tensor, np.ndarray, jaxtyping.Array)

XArray = ArrayAliasMeta("XArray", jaxtyping.Shaped, array_types=_tf_np_jnp)
