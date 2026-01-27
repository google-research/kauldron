# Copyright 2025 The kauldron Authors.
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

"""Provides a convenient way to access the current scope's dimensions."""

from __future__ import annotations

import enum

import immutabledict
from kauldron.ktyping import errors
from kauldron.ktyping import internal_typing
from kauldron.ktyping import scope as kscope

DimValue = internal_typing.DimValue
DimValues = internal_typing.DimValues
MISSING = internal_typing.MISSING


class DimView:
  """A view of the dimensions in the current scope.

  This is a user facing class that is used to access the dimensions in the
  current scope.
  It allows access to the known dimension values via dict-like access.

  Usage:
  ```

  kt.dim["a"] = 17  # sets the value of dim "a" to 17
  kt.dim["*b"] = (8,)  # sets the value of dim "*b" to (8,)

  batch_shape = kt.dim["*b"]  # e.g. (8,)

  del kt.dim["c"]  # removes the known values for dim "c"

  if "t" in kt.dim:  # checks if dim "t" has a known value in all candidates
    ...
  ```

  Note: this deliberateley abstracts away the fact that there might be multiple
  candidates in the current scope. The DimView will return dimension values,
  and allow setting dimensions values only if they are consistent across all
  candidates.
  It will raise errors if the requested dimension is not defined or ambiguous.
  I does allow deletion of dimensions, which will remove them from all
  candidates.
  """

  def __init__(self, scope: kscope.ShapeScope):
    self._scope = scope

  def __getitem__(self, name: str) -> int | DimValue:
    """Returns the value of a dimension in the current scope."""
    __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

    dim_type, name = _get_dim_type(name)

    values = {alt.get(name, MISSING) for alt in self._scope.candidates}
    if values == {MISSING}:
      raise KeyError(f"Unknown dimension: {name}")
    if len(values) > 1:
      raise errors.AmbiguousDimensionError(name, values)
    val = values.pop()

    assert not isinstance(val, internal_typing.Missing)

    if dim_type == _DimType.SINGLE:
      if len(val) != 1:
        raise ValueError(
            f"Regular (non */+) dims ({name!r}) must have length 1. Got: {val}"
        )
      return val[0]
    if dim_type == _DimType.MULTI:
      # TODO(klausg): Should we do something about Unknown dims? Error? None?
      return val
    elif dim_type == _DimType.PLUS:
      if not val:
        raise ValueError(
            f"Plus dims ('+{name}') must have at least one dimension."
        )
      return val
    else:
      raise ValueError(f"Unknown dim type: {dim_type}")

  def __contains__(self, name: str) -> bool:
    """Returns True if the dimension is defined in all candidates."""
    __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

    values = {alt.get(name, MISSING) for alt in self._scope.candidates}
    if MISSING in values:
      return False
    return True

  def __setitem__(self, name: str, value: int | DimValue):
    __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

    # TODO(klausg): support broadcastable dims
    dim_type, name = _get_dim_type(name)
    if dim_type == _DimType.MULTI and not isinstance(value, tuple):
      raise ValueError(f"Multi-dims ({name!r}) must be assigned a tuple.")
    if dim_type == _DimType.PLUS and (
        not value or not isinstance(value, tuple)
    ):
      raise ValueError(
          f"Plus-dims ({name!r}) must be assigned a non-empty tuple. "
          f"Got: {value}"
      )
    if dim_type == _DimType.SINGLE:
      if not isinstance(value, int):
        raise ValueError(
            f"Single dims ({name!r}) must be assigned an int. Got: {value}"
        )
      value = (value,)

    current_values = {alt.get(name, MISSING) for alt in self._scope.candidates}
    incompatible_values = current_values - {value, MISSING}

    if incompatible_values:
      raise ValueError(
          f"Incompatible values for {name!r} with {current_values=}. Cannot be"
          f" assigned to {value}."
      )

    # This means all the known values are compatible with the new value.
    # I.e. either they are all the same as the new value, or they are all
    # missing.
    modified_candidates = [
        alt | {name: value} for alt in self._scope.candidates
    ]
    self._scope.candidates = modified_candidates
    return

  def __delitem__(self, name: str):
    __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

    _, name = _get_dim_type(name)

    new_candidates = []
    deleted_at_least_one = False
    for alt in self._scope.candidates:
      assert isinstance(alt, immutabledict.immutabledict)
      if name in alt:
        alt = alt.delete(name)
        deleted_at_least_one = True
      new_candidates.append(alt)

    if not deleted_at_least_one:
      raise KeyError(name)

    self._scope.candidates = new_candidates

  def __str__(self) -> str:
    __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

    all_dims = {dim for cand in self._scope.candidates for dim in cand}  # pylint: disable=g-complex-comprehension
    ambiguous_dims = {}
    unambiguous_dims = {}
    for d in all_dims:
      all_values = [cand.get(d, MISSING) for cand in self._scope.candidates]
      values = set(all_values) - {MISSING}
      if len(values) == 1:
        unambiguous_dims[d] = values.pop()
      else:
        ambiguous_dims[d] = all_values

    # Right-justify the dimension names (up to 8 chars)
    max_dim_name_len = min(max(len(dim) for dim in all_dims) + 1, 8)
    # Format unambiguous dims like this:
    # dims = {
    #   *b: (8, 32)
    #    d: 7
    #    y: #
    # }
    dims_str = "dims = {\n"
    if unambiguous_dims:
      dims_str += "\n" + "\n".join(
          _format_dim_assignment(dim, value, align=max_dim_name_len)
          for dim, value in unambiguous_dims.items()
      )
    dims_str += "\n}\n"

    # Format ambiguous dims (if any) like this:
    # ambiguous_dims = {
    #   *b: (8, 32) | (8, 16)
    #    d: 7 | 17
    #    y: # | 1
    # }
    if ambiguous_dims:
      dims_str += (
          "ambiguous_dims = {\n"
          + "\n".join(
              _format_ambiguous_dim(dim, values, align=max_dim_name_len)
              for dim, values in ambiguous_dims.items()
          )
          + "\n}\n"
      )

    return dims_str


def _format_dim_value(value: DimValue) -> str:
  str_values = [str(v) if isinstance(v, int) else "#" for v in value]
  if len(value) == 1:
    return str_values[0]
  else:
    return f"({', '.join(str_values)})"


def _format_dim_assignment(dim_name: str, value: DimValue, align: int) -> str:
  if len(value) == 1:
    dim_name = dim_name.rjust(align)
  else:
    dim_name = f"*{dim_name}".rjust(align)

  return f"  {dim_name}: {_format_dim_value(value)}"


def _format_ambiguous_dim(
    dim_name: str, values: list[DimValue], align: int
) -> str:
  is_tuple = any(len(v) > 1 for v in values)

  if is_tuple:
    dim_name = f"*{dim_name}".rjust(align)
  else:
    dim_name = dim_name.rjust(align)

  values_str = " | ".join(_format_dim_value(v) for v in values)
  return f"  {dim_name}: {values_str}"


class _DimType(enum.Enum):
  """Represents the type of a dimension."""

  SINGLE = enum.auto()
  MULTI = enum.auto()
  PLUS = enum.auto()


def _get_dim_type(name: str) -> tuple[_DimType, str]:
  if name[0] == "*":
    return _DimType.MULTI, name[1:]
  elif name[0] == "+":
    return _DimType.PLUS, name[1:]
  else:
    return _DimType.SINGLE, name


class _CurrentDimView(DimView):
  """Convenience class to access the current scope's DimView."""

  def __init__(self):  # pylint: disable=super-init-not-called
    pass

  @property
  def _scope(self) -> kscope.ShapeScope:
    __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable
    return kscope.get_current_scope(nested_ok=False)


# A convenience alias for the current scope's DimView.
# use like this:
# import kauldron.ktyping as kt
#
# with kt.typechecked():  # open a scope
#   kt.dim["a"] = 17
dim = _CurrentDimView()
