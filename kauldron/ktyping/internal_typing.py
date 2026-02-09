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

"""Internal types."""

import enum
import typing
import immutabledict


class Missing(enum.Enum):
  """Special value for missing arguments and variables."""

  MISSING = enum.auto()


class UnknownDim(enum.Enum):
  """Special value for unknown dimensions."""

  UNKNOWN_DIM = enum.auto()


MISSING = Missing.MISSING
UNKNOWN_DIM = UnknownDim.UNKNOWN_DIM


Shape: typing.TypeAlias = tuple[int, ...]


# DimValue corresponds to the known value of a dimension.
# It is either a tuple of integers, or a special value for unknown dimensions.
# The latter can happen for broadcastable multi-dims where the observed value
# for a dimension was 1. In that case the value of the dimension could be 1 or
# any other value.
DimValue: typing.TypeAlias = tuple[int | UnknownDim, ...]

# Store the mappings of dim names to known values.
DimValues: typing.TypeAlias = immutabledict.immutabledict[str, DimValue]

# CandidateDims is a set possible dimension <-> value mappings.
# It is used to represent the set of possible dim values that are compatible
# with the observed shapes and annotations in a given scope.
# This can be more than one if the annotations allow multiple different
# assignments of dimensions (usually because of a Union).
CandidateDims: typing.TypeAlias = frozenset[DimValues]
