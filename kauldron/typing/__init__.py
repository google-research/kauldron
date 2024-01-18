# Copyright 2023 The kauldron Authors.
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

"""Common Typing Annotations."""
from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence, Union

from clu.data.dataset_iterator import ArraySpec, ElementSpec, PyTree  # pylint: disable=g-multiple-import,g-importing-member
import jax
from kauldron.typing.array_types import (  # pylint: disable=g-multiple-import
    Array,
    Bool,
    Float,
    Float32,
    Int,
    Integer,
    Scalar,
    ScalarFloat,
    ScalarInt,
    TfArray,
    TfFloat,
    TfFloat32,
    TfInt,
    TfUInt8,
    UInt32,
    UInt8,
)
from kauldron.typing.shape_spec import Memo, Shape  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.typing.type_check import TypeCheckError, check_type, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np

PRNGKey = UInt32["2"]
PRNGKeyLike = Union[int, Sequence[int], np.ndarray, PRNGKey]

Dtype = Any

Initializer = Callable[[PRNGKey, Shape, Dtype], Array]
Axes = int | tuple[int, ...]
AxisName = Hashable
Schedule = Callable[[int], float | Float[""]]
