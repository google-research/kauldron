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

"""Dump of errors raised by ktyping."""
from __future__ import annotations

from collections.abc import Collection
import inspect
import typing
from typing import Any

from kauldron.ktyping import internal_typing
from kauldron.ktyping import utils
import typeguard

if typing.TYPE_CHECKING:
  from kauldron.ktyping import scope as kscope  # pylint: disable=g-bad-import-order


MISSING = internal_typing.MISSING
TypeCheckError = typeguard.TypeCheckError


# MARK: AmbiguousDim
class AmbiguousDimensionError(KeyError):
  """Raised when a dimension is ambiguous within the current scope.

  For example in the following code, the dimension 'b' is ambiguous because both
  annotations in the union are valid:
  ```
  import kauldron.ktyping as kt

  @kt.typechecked
  def f(x: kt.Float["*b"] | kt.Float["*b n"]):
    print(kt.dim["*b"])

  f(np.zeros((2, 3)))
  # AmbiguousDimensionError: Dimension 'b' is ambiguous within the current
  # scope. Could be one of: {(2, 3), (2,)}
  ```
  """

  def __init__(
      self, name: str, candidates: Collection[internal_typing.DimValues]
  ):
    message = (
        f"Dimension '{name}' is ambiguous within the current scope. Could be"
        f" one of: {candidates!r}"
    )
    super().__init__(message)


# MARK: KTypeCheckErr
class KTypeCheckError(TypeCheckError):
  """Raised when a ktyping type check fails."""

  def __init__(
      self,
      message: str,
      scope: kscope.ShapeScope,
  ):
    super().__init__(message)
    # Code location information
    self.definition_source = scope.source
    self.entry_source = scope.active_scope.source
    self.highlight = None
    self.arg_info = scope.get_arg_info()
    self.candidates = scope.candidates
    self.return_annotation = scope.annotations.get(
        "return", inspect.Parameter.empty
    )
    self.return_value = scope.return_value

  @classmethod
  def raise_from_exc(
      cls,
      exc: TypeCheckError,
      scope: kscope.ShapeScope,
      additional_path_element: str | None = None,
      maybe_highlight: str | None = None,
  ):
    # Hide the function from the traceback. Supported by Pytest and IPython
    __tracebackhide__ = True  # pylint: disable=invalid-name,unused-variable

    if isinstance(exc, KTypeCheckError):
      ktyping_exc = exc
    else:
      ktyping_exc = cls(exc.args[0], scope)
      ktyping_exc._path = exc._path  # pylint: disable=protected-access

    if additional_path_element is not None:
      ktyping_exc.append_path_element(additional_path_element)
    if maybe_highlight is not None:
      ktyping_exc.maybe_set_highlight(maybe_highlight)
    raise ktyping_exc.with_traceback(exc.__traceback__) from exc.__cause__

  def maybe_set_highlight(self, highlight: str | None = None):
    # Only remember the first highlight, because that points to the argument
    # that caused the error.
    if self.highlight is None:
      self.highlight = highlight

  @property
  def origin_block(self) -> str:
    if self.definition_source == self.entry_source:
      return f"Origin: {self.definition_source.to_str()}"
    else:
      return (
          f"Origin: {self.definition_source.to_str()}\n"
          f"        uses scope from {self.entry_source.to_str()}"
      )

  @property
  def arguments_block(self) -> str:
    arg_lines = ["Arguments:"]
    has_default_args = False
    # First, all non-default arguments.
    for name, (value, annot, is_default) in self.arg_info.items():
      if is_default:
        has_default_args = True
        continue
      arg_lines.append(self._format_arg_line(name, value, annot))
    # If there are default arguments, add a blank line before listing them.
    if has_default_args:
      arg_lines.append("")
      for name, (value, annot, is_default) in self.arg_info.items():
        if not is_default:
          continue
        arg_lines.append(self._format_arg_line(name, value, annot))

    return "\n".join(arg_lines)

  @property
  def return_block(self) -> str:
    if self.return_annotation == inspect.Parameter.empty:
      return ""
    annot_str = utils.get_type_name(self.return_annotation)
    if self.return_value is MISSING:
      return f"Return: {annot_str}"
    else:
      return f"Return: {annot_str} = {utils.format_value(self.return_value)}"

  @property
  def candidates_block(self) -> str:
    """Formats the scope section of an error message."""
    if len(self.candidates) == 1:
      scope_lines = ["Dim Assignments:"]
    else:
      scope_lines = ["Multiple Dim Assignment Candidates:"]

    for candidate in self.candidates:
      dim_assignments = [
          _format_dim_assignment(dim, value)
          for dim, value in candidate.items()
          if not internal_typing.is_structure_key(dim)
      ]
      dims = ", ".join(dim_assignments)
      scope_lines.append(f" - {{{dims}}}")

    tree_structures = {}
    for candidate in self.candidates:
      for key, value in candidate.items():
        if internal_typing.is_structure_key(key):
          tree_structures[key] = value
    if tree_structures:
      scope_lines.append("Tree Structures:")
      for key, value in tree_structures.items():
        # TODO(klausg): abbreviated display for long treedefs
        scope_lines.append(f"  {key}: {value}")

    return "\n".join(scope_lines)

  def _format_arg_line(self, name, value, annot):
    if annot != inspect.Parameter.empty:
      annot_str = f": {utils.get_type_name(annot)}"
    else:
      annot_str = ""
    indent = "> " if self.highlight == name else "  "
    return f"{indent}{name}{annot_str} = {value}"

  def __str__(self):
    message = super().__str__()
    # Remove the namespace of the array type meta classes for readability.
    message = message.replace("kauldron.ktyping.array_type_meta.", "")
    return "\n\n".join([
        message,
        self.origin_block,
        self.arguments_block,
        self.return_block,
        self.candidates_block,
    ])


def _format_dim_assignment(dim, value):
  if len(value) == 1:
    return f"{dim}: {value[0]}"
  else:
    return f"*{dim}: {value}"


# MARK: Error messages
def array_type_error_message(
    value: Any,
    acceptable_array_types: Collection[type[Any]],
    type_spec: Any | None = None,
) -> str:
  """Returns an error message for array type errors."""
  array_spec_str = f" (required by {type_spec!r})" if type_spec else ""
  array_type_str = utils.get_type_name(value, full_path=True)
  array_types_str = "|".join(
      sorted({utils.get_type_name(a) for a in acceptable_array_types})
  )
  return (
      f"was of type {array_type_str} which is not an instance of"
      f" {array_types_str}{array_spec_str}."
  )


def dtype_error_message(
    value: Any,
    acceptable_dtypes: Collection[Any],
    type_spec: Any | None = None,
) -> str:
  """Returns an error message for dtype errors."""
  array_spec_str = f" (required by {type_spec!r})" if type_spec else ""
  dtypes_str = utils.get_dtype_str(value)
  acceptable_dtypes_str = "|".join(
      sorted({str(dtype) for dtype in acceptable_dtypes})
  )
  return (
      f"has dtype {dtypes_str!r} which is not dtype-compatible with "
      f"{acceptable_dtypes_str}{array_spec_str}"
  )


def shape_error_message(
    value: Any,
    acceptable_shapes: Collection[str],
    type_spec: Any | None = None,
) -> str:
  """Returns an error message for shape errors."""
  array_spec_str = f" (required by {type_spec!r})" if type_spec else ""
  acceptable_shapes = sorted(set(acceptable_shapes))
  if len(acceptable_shapes) == 1:
    return (
        f"has shape {value.shape} which is not shape-compatible with"
        f" {acceptable_shapes[0]!r}{array_spec_str}."
    )
  else:
    return (
        f"has shape {value.shape} which is not shape-compatible with any of"
        f" {acceptable_shapes!r}{array_spec_str}."
    )
