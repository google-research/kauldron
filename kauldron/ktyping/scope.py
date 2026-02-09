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

"""Defines the DimScope class and associated functions.

This class is one of the central components of ktyping.
It is used to define a scope and keep track of the available shape information.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import inspect
import types
from typing import Any, Iterable, Self

from etils import edc
from etils import epy
from kauldron.ktyping import config
from kauldron.ktyping import dim_view
from kauldron.ktyping import frame_utils
from kauldron.ktyping import internal_typing
from kauldron.ktyping import log
from kauldron.ktyping import utils

MISSING = internal_typing.MISSING
DimValue = internal_typing.DimValue
DimValues = internal_typing.DimValues
CandidateDims = internal_typing.CandidateDims


# MARK: ShapeScope
class ShapeScope(epy.ContextManager):
  """Scope objects are used to keep track of the available shape information.

  Can be used as a context manager to open a new scope.
  Is used in the @typechecked decorator to open a new scope for each
  @typechecked function.

  Attributes:
    source: The origin (description, file, line number) of where the scope was
      opened. I.e. the function definition, the context manager, the TypedDict,
      ...
    candidates: The current set of candidate dim values in this scope.
    arguments: The arguments of the function if the scope was opened by a
      typechecked function, otherwise None.
    annotations: The annotations of the function if the scope was opened by a
      typechecked function, otherwise None.
    fstring_locals: The namespace used for fstring evaluation. E.g. for
      typechecked functions this is the bound arguments of the function call.
    default_args: A list of argument names that still have their default value.
      (only useful in combination with arguments)
    stacklevel: The stack level of the caller frame to check for an active
      scope. Used internally e.g. for the typechecked context manager.
  """

  def __init__(
      self,
      source: utils.CodeLocation = utils.CodeLocation.unknown(),
      candidates: Sequence[Mapping[str, DimValue]] | None = None,
      arguments: Mapping[str, Any] | None = None,
      annotations: Mapping[str, Any] | None = None,
      fstring_locals: Mapping[str, Any] | None = None,
      default_args: Sequence[str] | None = None,
      stacklevel: int = 0,
  ):
    self._candidates: CandidateDims = frozenset([DimValues()])
    self._stacklevel = stacklevel
    self.source = source
    if candidates is not None:
      # use setter to normalize the candidates
      self.candidates = candidates
    self.arguments = arguments if arguments is not None else {}
    self.annotations = annotations if annotations is not None else {}
    self.fstring_locals = fstring_locals if fstring_locals is not None else {}
    self.default_args = default_args or set()
    self.return_value = MISSING

    self._check_for_jaxtyping_annotations()

  @property
  def active_scope(self) -> Self:
    """Returns the first non-transparent scope, in this case just self."""
    return self

  @property
  def candidates(self) -> CandidateDims:
    return self._candidates

  @candidates.setter
  def candidates(self, new_value: Sequence[Mapping[str, DimValue]]):
    self._candidates = frozenset([DimValues(alt) for alt in new_value])

  @functools.cached_property
  def dim(self) -> dim_view.DimView:
    return dim_view.DimView(self)

  def get_arg_info(self) -> Mapping[str, tuple[str, Any, bool]]:
    """Returns a mapping from argument name to (repr, annot, is_default)."""
    return {
        name: (
            utils.format_value(value),
            self.annotations.get(name, inspect.Parameter.empty),
            name in self.default_args,
        )
        for name, value in self.arguments.items()
    }

  def __contextmanager__(self) -> Iterable[Self]:
    __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

    # Push self onto the stack
    _scope_stack.append(self)
    # stacklevel + 2 to ignore the double indirection by epy.ContextManager
    caller_frame = frame_utils.get_caller_frame(stacklevel=self._stacklevel + 2)
    frame_utils.mark_frame_as_active_scope(caller_frame)
    try:
      yield self
    finally:
      frame_utils.unmark_frame_as_active_scope(caller_frame)
      s = _scope_stack.pop()
      assert s == self

  def _check_for_jaxtyping_annotations(self) -> None:
    """Warns about non-ktyping types in the annotations to help migrations."""
    reporting_policy = config.get_config(self.source).jaxtyping_annotations
    for name, annot in self.annotations.items():
      if utils.contains_jaxtyping_type(annot):
        log.report(
            "Mixing of ktyping and jaxtyping detected: Found jaxtyping"
            f" annotation in ktyping scope at {self.source.to_str()} for"
            f" {name}: {utils.get_type_name(annot)}",
            policy=reporting_policy,
        )


# MARK: TransparentScope
class TransparentScope(ShapeScope):
  """A transparent scope that forwards all calls to the parent scope."""

  def __init__(
      self,
      source: utils.CodeLocation = utils.CodeLocation.unknown(),
      candidates: Sequence[Mapping[str, DimValue]] | None = None,
      arguments: Mapping[str, Any] | None = None,
      annotations: Mapping[str, Any] | None = None,
      fstring_locals: Mapping[str, Any] | None = None,
      default_args: Sequence[str] | None = None,
      stacklevel: int = 0,
  ):
    assert candidates is None
    super().__init__(
        source=source,
        arguments=arguments,
        annotations=annotations,
        fstring_locals=fstring_locals,
        default_args=default_args,
        stacklevel=stacklevel,
    )
    self._parent_scope = None

  @property
  def active_scope(self) -> Self:
    """Returns the first non-transparent scope."""
    if self._parent_scope is None:
      raise RuntimeError("TransparentScope cannot be used as top level scope.")
    return self._parent_scope.active_scope

  @property
  def candidates(self) -> CandidateDims:
    return self.active_scope.candidates  # pytype: disable=bad-return-type

  @candidates.setter
  def candidates(self, new_value: Sequence[Mapping[str, DimValue]]):
    self.active_scope.candidates = new_value

  def __enter__(self) -> Self:
    if not _scope_stack:
      raise frame_utils.NoActiveScopeError()
    self._parent_scope = _scope_stack[-1]
    return super().__enter__()

  def __exit__(self, exc_type, exc_value, traceback):
    self._parent_scope = None
    return super().__exit__(exc_type, exc_value, traceback)


# MARK: scope management

# A thread-local (and coroutine-local) stack of Scope objects.
# It is used to keep track of the available shape information for the current
# scope. Usually opened by the @typechecked decorator.
_scope_stack = edc.ContextStack[ShapeScope]()


def create_scope_for(
    obj: types.FunctionType | type[Any],
    arguments: Mapping[str, Any] | None = None,
    annotations: Mapping[str, Any] | None = None,
    default_args: Sequence[str] | None = None,
    fstring_locals: Mapping[str, Any] | None = None,
    transparent: bool | None = None,
    source: utils.CodeLocation | None = None,
) -> ShapeScope:
  """Returns a new ShapeScope for a given function or class."""
  if source is None:
    source = utils.CodeLocation.from_any(obj)
  ScopeClass = TransparentScope if transparent else ShapeScope
  return ScopeClass(
      source=source,
      arguments=arguments,
      annotations=annotations,
      fstring_locals=fstring_locals,
      default_args=default_args,
  )


def get_current_scope(
    nested_ok: bool = False, stacklevel: int = 0
) -> ShapeScope:
  """Returns the current scope.

  Args:
    nested_ok: By default this function will raise an error if the caller is not
      itself a @typechecked function. Disable this check by passing
      nested_ok=True. This allows accessing the currently active scope from from
      a non-@typechecked function.
    stacklevel: The stack level of the caller frame to check for an active
      scope. Only used if nested_ok is False.

  Returns:
    The currently active scope.

  Raises:
    NoActiveScopeError: If there is no active scope, or if the caller is not
    @typechecked and nested_ok is False.
  """
  __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

  if not nested_ok:
    frame_utils.assert_caller_has_active_scope(stacklevel=stacklevel)

  if not _scope_stack:
    raise frame_utils.NoActiveScopeError()

  return _scope_stack[-1]


def is_scope_stack_empty() -> bool:
  """Returns True if the scope stack is empty."""
  return not bool(_scope_stack)
