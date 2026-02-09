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

"""Ktyping: A library for type annotations of arrays."""

# pylint: disable=g-multiple-import, g-importing-member
# fmt: skip-import-sorting
from typing import Sequence
import jax.typing
from kauldron.ktyping import dtypes
from kauldron.ktyping.array_type_meta import is_array_type
from kauldron.ktyping.array_types import (
    # jax / numpy
    Array,
    BFloat16,
    Bool,
    Complex,
    Complex128,
    Complex64,
    Float,
    Float16,
    Float32,
    Float64,
    Int,
    Int16,
    Int32,
    Int64,
    Int8,
    IntOrBool,
    Num,
    PRNGKey,
    PRNGKeyArray,
    SInt,
    UInt,
    UInt16,
    UInt32,
    UInt64,
    UInt8,
    # scalars
    Scalar,
    ScalarBool,
    ScalarComplex,
    ScalarFloat,
    ScalarInt,
    # tensorflow
    TfArray,
    TfBool,
    TfComplex,
    TfComplex128,
    TfComplex64,
    TfFloat,
    TfFloat16,
    TfFloat32,
    TfFloat64,
    TfInt,
    TfInt16,
    TfInt32,
    TfInt64,
    TfInt8,
    TfIntOrBool,
    TfNum,
    TfSInt,
    TfUInt,
    TfUInt16,
    TfUInt32,
    TfUInt64,
    TfUInt8,
    # any array type
    XArray,
    XBool,
    XComplex,
    XComplex128,
    XComplex64,
    XFloat,
    XFloat16,
    XFloat32,
    XFloat64,
    XInt,
    XInt16,
    XInt32,
    XInt64,
    XInt8,
    XIntOrBool,
    XNum,
    XSInt,
    XUInt,
    XUInt16,
    XUInt32,
    XUInt64,
    XUInt8,
    # shape types
    Shape,
)
from kauldron.ktyping.config import (
    CONFIG,
    Config,
    add_config_override,
    remove_config_override,
)
from kauldron.ktyping.decorator import typechecked
from kauldron.ktyping.dim_view import dim
from kauldron.ktyping.errors import AmbiguousDimensionError
from kauldron.ktyping.errors import KTypeCheckError
from kauldron.ktyping.frame_utils import NoActiveScopeError
from kauldron.ktyping.pytree import PyTree
from kauldron.ktyping.scope import get_current_scope
from kauldron.ktyping.scope import ShapeScope
from kauldron.ktyping.shape_tools import shape
from kauldron.ktyping.typeguard_checkers import check_type
# pylint: enable=g-multiple-import, g-importing-member

# Permissive dtype annotation that accepts:
# * strings like 'float32', 'int32'
# * types like np.float32, np.int32, float, int
# * np.dtype objects like np.dtype('float32'), np.dtype('int32')
# * jax dtypes like jnp.float32, jnp.int32
DType = jax.typing.DTypeLike

# Annotation for axes arguments (e.g. for `sum(array, axis=(1, 2))`)
Axes = int | Sequence[int]
