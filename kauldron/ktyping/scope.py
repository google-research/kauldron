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

"""Defines the DimScope class and associated functions.

This class is one of the central components of ktyping.
It is used to define a scope and keep track of the available shape information.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import inspect
from typing import Any, Callable, Self
import weakref

from etils import edc
from kauldron.ktyping import errors
from kauldron.ktyping import utils
from kauldron.ktyping.constraints import Constraint, ConstraintAlternatives, Constraints  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.ktyping.internal_typing import UNDEFINED, Undefined  # pylint: disable=g-multiple-import,g-importing-member


DIM_SCOPE_REF = "__ktyping_scope__"


class ShapeScope:
  """Scope objects are used to keep track of the available shape information.

  Can be used as a context manager to open a new scope.
  Is used in the @typechecked decorator to open a new scope for each
  @typechecked function.

  TODO(klausg): move the shape() function to this class?
  """

  def __init__(
      self,
      function: Callable[..., Any] | None = None,
      bound_args: Mapping[str, Any] | None = None,
      alternatives: Sequence[Mapping[str, Constraint]] | None = None,
  ):
    self.caller_frame = None
    self.function = function
    self.bound_args = bound_args
    self._alternatives: ConstraintAlternatives = frozenset([Constraints()])
    if alternatives is not None:
      # use setter to normalize the alternatives
      self.alternatives = alternatives

  @property
  def alternatives(self) -> ConstraintAlternatives:
    return self._alternatives

  @alternatives.setter
  def alternatives(self, new_value: Sequence[Mapping[str, Constraint]]):
    self._alternatives = frozenset([Constraints(alt) for alt in new_value])

  @functools.cached_property
  def dims(self) -> DimView:
    return DimView(self)

  def __enter__(self) -> Self:
    # Push self onto the stack
    _scope_stack.append(self)
    self._caller_frame = _get_caller_frame(offset=0)
    self._add_dim_scope_ref_to_caller_locals()
    return self

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    # TODO(klausg): add error information to the exception if possible
    self._remove_dim_scope_ref_from_caller_locals()
    self._caller_frame = None
    s = _scope_stack.pop()
    assert s == self

  def _add_dim_scope_ref_to_caller_locals(self) -> None:
    """Adds a dim scope ref as a marker to the caller's local variables.

    This can be used later to check if caller has an active DimScope.
    """
    assert self._caller_frame is not None
    ref = self._caller_frame.frame.f_locals.get(DIM_SCOPE_REF, None)
    # TODO(klausg): should we allow nested scopes in the same frame?
    if ref is not None and ref() is not None:
      raise ValueError(
          "DimScope is already active in this scope. "
          "Did you forget to close the previous scope?"
      )
    self._caller_frame.frame.f_locals[DIM_SCOPE_REF] = weakref.ref(self)

  def _remove_dim_scope_ref_from_caller_locals(self) -> None:
    """Adds a dim scope ref as a marker to the caller's local variables."""
    assert self._caller_frame is not None
    ref = self._caller_frame.frame.f_locals.get(DIM_SCOPE_REF, None)
    if ref is None and not utils.is_running_in_debugger():
      raise AssertionError("No active DimScope.")
    # TODO(klausg): maybe check if ref points to correct DimScope?
    del self._caller_frame.frame.f_locals[DIM_SCOPE_REF]


# A thread-local (and coroutine-local) stack of Scope objects.
# It is used to keep track of the available shape information for the current
# scope. Usually opened by the @typechecked decorator.
_scope_stack = edc.ContextStack[ShapeScope]()


def has_active_scope() -> bool:
  """Returns True if there is an active scope."""
  return bool(_scope_stack)


def get_current_scope(subscope_ok: bool = False) -> ShapeScope:
  """Returns the current scope."""
  if not subscope_ok:
    assert_caller_has_active_scope()

  if not _scope_stack:
    raise errors.NoActiveScopeError()

  return _scope_stack[-1]


def assert_caller_has_active_scope() -> None:
  """Raises AssertionError if the calling function is not @typechecked."""
  # Make sure we are not running in a debugger, because this check would break.
  if utils.is_running_in_debugger():
    return

  frame = _get_caller_frame(offset=1)  # offset=1: get caller of our caller
  try:
    dims_ref = frame.frame.f_locals.get(DIM_SCOPE_REF, None)
    if not dims_ref or dims_ref() is None:
      raise errors.NoActiveScopeError()  # TODO(klausg): add a message
  finally:
    # Make sure we do not leave unnecessary reference cycles behind.
    del frame


class DimView:
  """A view of the dimensions in the current scope.

  This is a user facing class that is used to access the dimensions in the
  current scope.
  It allows access to the known dimension values via dict-like and
  attribute-like access.

  Note: this deliberateley abstracts away the fact that there might be multiple
  alternatives in the current scope. The DimView will return dimension values,
  and allow setting dimensions values only if they are consistent across all
  alternatives.
  It will raise errors if the requested dimension is not defined or ambiguous.
  I does allow deletion of dimensions, which will remove them from all
  alternatives.
  """

  def __init__(self, scope: ShapeScope):
    object.__setattr__(self, "_scope", scope)

  def __getitem__(self, name: str) -> int | Undefined | Constraint:
    """Returns the value of a dimension in the current scope."""
    dim_type, name = _get_dim_type(name)

    values = {alt.get(name, UNDEFINED) for alt in self._scope.alternatives}
    if values == {UNDEFINED}:
      raise errors.UnknownDimensionError(name)
    if len(values) > 1:
      raise errors.AmbiguousDimensionError(name, values)
    val = values.pop()

    assert not isinstance(val, Undefined)

    if dim_type == "single":
      if len(val) != 1:
        raise errors.DimLengthError(name, val)
      return val[0]
    if dim_type == "multi":
      # TODO(klausg): Should we do something about Unknown dims? Error? None?
      return val
    elif dim_type == "plus":
      if not val:
        raise errors.DimLengthError(
            name, val
        )  # TODO(klausg): better error message
      return val
    else:
      raise ValueError(f"Unknown dim type: {dim_type}")

  def __setitem__(self, name: str, value: int | Constraint):
    # TODO(klausg): support broadcastable dims
    dim_type, name = _get_dim_type(name)
    if dim_type == "multi" and not isinstance(value, tuple):
      raise ValueError(f"Multi-dims ({name!r}) must be assigned a tuple.")
    if dim_type == "plus" and (not value or not isinstance(value, tuple)):
      raise ValueError(
          f"Plus-dims ({name!r}) must be assigned a non-empty tuple. "
          f"Got: {value}"
      )
    if dim_type == "single":
      if not isinstance(value, int):
        raise ValueError(
            f"Single dims ({name!r}) must be assigned an int. Got: {value}"
        )
      value = (value,)

    current_values = {
        alt.get(name, UNDEFINED) for alt in self._scope.alternatives
    }
    incompatible_values = current_values - {value, UNDEFINED}

    if incompatible_values:
      raise errors.IncompatibleDimensionError(name, incompatible_values)

    # This means all the known values are compatible with the new value.
    # I.e. either they are all the same as the new value, or they are all
    # undefined.
    modified_alternatives = [
        alt | {name: value} for alt in self._scope.alternatives
    ]
    self._scope.alternatives = modified_alternatives
    return

  def __delitem__(self, name: str):
    _, name = _get_dim_type(name)

    new_alternatives = []
    deleted_at_least_one = False
    for alt in self._scope.alternatives:
      assert isinstance(alt, Constraints)
      if name in alt:
        alt = alt.delete(name)
        deleted_at_least_one = True
      new_alternatives.append(alt)

    if not deleted_at_least_one:
      raise KeyError(name)

    self._scope.alternatives = new_alternatives

  def __getattr__(self, k: str) -> Any:
    try:
      return object.__getattribute__(self, k)
    except AttributeError:
      return self[k]

  def __setattr__(self, k: str, v: Any):
    try:
      # Throws exception if not in prototype chain
      object.__getattribute__(self, k)
    except AttributeError:
      self[k] = v
    else:
      object.__setattr__(self, k, v)

  def __delattr__(self, k: str):
    try:
      # Throws exception if not in prototype chain
      object.__getattribute__(self, k)
    except AttributeError:
      del self[k]
    else:
      object.__delattr__(self, k)


class TransparentScope(ShapeScope):
  """A transparent scope that forwards all calls to the parent scope."""

  def __init__(
      self,
      *,
      fn: Callable[..., Any] | None = None,
      bound_args: Mapping[str, Any] | None = None,
  ):
    super().__init__(fn, bound_args)
    self._parent_scope = None

  @property
  def alternatives(self) -> ConstraintAlternatives:
    if self._parent_scope is None:
      raise RuntimeError(
          "No parent scope."
      )  # TODO(klausg): better error message
    return self._parent_scope.alternatives  # pytype: disable=bad-return-type

  @alternatives.setter
  def alternatives(self, new_value: Sequence[Mapping[str, Constraint]]):
    if self._parent_scope is None:
      raise RuntimeError(
          "No parent scope."
      )  # TODO(klausg): better error message
    self._parent_scope.alternatives = new_value

  def __enter__(self) -> Self:
    # Push self onto the stack
    if not _scope_stack:
      raise errors.NoActiveScopeError()  # TODO(klausg): better error message
    self._parent_scope = _scope_stack[-1]
    return super().__enter__()

  def __exit__(self, exc_type, exc_value, traceback) -> None:
    super().__exit__(exc_type, exc_value, traceback)
    self._parent_scope = None


LAMBDA_AND_COMPREHENSIONS = frozenset(
    ("<lambda>", "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>")
)


def _get_caller_frame(
    ignore: frozenset[str] = LAMBDA_AND_COMPREHENSIONS,
    offset: int = 0,
) -> inspect.FrameInfo:
  """Returns FrameInfo for the caller fun while ignoring comprehensions etc."""
  stack = inspect.stack()
  try:
    # stack[0].function = _get_caller_frame
    # ...
    # stack[1 + offset] = function for which we wants to get the caller frame
    # ... possibly ignored frames, e.g. due to lambdas or comprehensions.
    # stack[i]  <- the intended caller frame
    i = 2 + offset
    while stack[i].function in ignore and i < len(stack) - 2:
      i += 1
    return stack[i]
  finally:
    # Make sure we do not leave unnecessary reference cycles behind.
    del stack


def _get_dim_type(name: str) -> tuple[str, str]:
  if name[0] == "*":
    return "multi", name[1:]
  elif name[0] == "+":
    return "plus", name[1:]
  else:
    return "single", name
