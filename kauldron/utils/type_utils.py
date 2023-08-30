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

"""Typing annotation utils."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import types
import typing
from typing import Any, Annotated, TypeVar, Union

from etils import epy
import typing_extensions

# Might be replaced in the future: https://github.com/python/mypy/issues/9773
_TypeForm = Any
_AnnotatedType = Any  # Like `TypeForm[Annoted[_T]]`

_Token = object

_FnT = TypeVar('_FnT')
_SelfT = TypeVar('_SelfT')


def get_annotated(
    cls_or_obj: type[Any] | Any,
    annotated_token: _Token | _AnnotatedType,
) -> list[str]:
  """Returns all the attributes names annotated with `annotated_token`.

  Examples:

  ```python
  annotated_token = object()

  class A:
    a: Annotated[str, annotated_token]
    b: Annotated[int, annotated_token, other_token]
    c: Annotated[int, other_token]
    d: int

  get_annotated(A, annotated_token) == ['a', 'b']
  ```

  Args:
    cls_or_obj: Class or instance on which compute the annotations
    annotated_token: The token of the annotated. `Annotated[Any, token]` is also
      accepted

  Returns:
    hints:
  """
  cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
  # The first time, compute typing annotations & metadata
  # At this point, `ForwardRef` should have been resolved.
  hints = _get_type_hints(cls)

  # Unwrap the token if `Annotated[_T, token]` is given.
  if _is_annotated_type(annotated_token):
    (annotated_token,) = annotated_token.__metadata__  # pytype: disable=attribute-error

  # Filter all hints annotated with the token
  return [
      name
      for name, hint in hints.items()
      if _is_hint_annotated_with(hint, annotated_token)
  ]


def get_optional_fields(cls_or_obj: type[Any] | Any) -> list[str]:
  cls = cls_or_obj if isinstance(cls_or_obj, type) else type(cls_or_obj)
  return [n for n, a in _get_type_hints(cls).items() if _is_optional(a)]


def _is_optional(annotation: type[Any]) -> bool:
  """Check if a typing annotation is Optional[...]."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)
  return origin is Union and type(None) in args


def _is_hint_annotated_with(hint: _TypeForm, annotated_token: _Token) -> bool:
  """Returns `True` if the hint is annotated with `annotated_token`."""
  type_visitor = _AnnotatedCheckVisitor(annotated_token)
  return type_visitor.visit(hint).token_present


# TODO(epot): Should try to unify with:
# * https://github.com/google-research/dataclass_array/tree/HEAD/dataclass_array/type_parsing.py
# * https://github.com/google/etils/tree/HEAD/etils/enp/type_parsing.py
class _TypeVisitor:
  """Traverse the tree of typing annotations.

  Usage:

  ```python
  @dataclasses.dataclass
  class MyVisitor(_TypeVisitor):
    leaves = dataclasses.field(default_factory=list)

    def _visit_leaf(self, hint):
      self.leaves.append(hint)


  MyVisitor().visit(int | list[str]).leaves == [int, str]
  ```
  """

  @functools.cached_property
  def _origin_to_visitor(self) -> dict[_TypeForm, Callable[[_TypeForm], None]]:
    """Mapping `__origin__` to visitor."""
    return {
        typing.Annotated: self._visit_annotated,
        typing.Union: self._visit_union,
        types.UnionType: self._visit_union,
        None: self._visit_leaf,  # Default origin
    }

  def visit(self: _SelfT, hint: _TypeForm) -> _SelfT:
    """Traverse the tree of types."""
    if hint == types.NoneType:  # Normalize `None`
      hint = None

    origin = typing_extensions.get_origin(hint)
    visit_fn = self._origin_to_visitor.get(origin, self._visit_leaf)  # pylint: disable=protected-access
    visit_fn(hint)
    return self

  def _visit_union(self, hint: _TypeForm) -> None:
    """Traverse `T0 | T1` and `Optional[T]`."""
    inner_hints = typing_extensions.get_args(hint)
    for inner_hint in inner_hints:
      self.visit(inner_hint)

  def _visit_annotated(self, hint: _TypeForm) -> None:
    """Traverse `Annotated[T]`."""
    inner_hint, *_ = typing_extensions.get_args(hint)
    self.visit(inner_hint)

  def _visit_leaf(self, hint: _TypeForm) -> None:
    """Leaves nodes."""
    pass


@dataclasses.dataclass
class _AnnotatedCheckVisitor(_TypeVisitor):
  """Type visitor which check the token is present in the tree."""
  annotated_token: _Token
  token_present: bool = False

  def _visit_annotated(self, hint: _TypeForm):
    super()._visit_annotated(hint)
    if any(a is self.annotated_token for a in hint.__metadata__):
      self.token_present = True


# This could be removed in Python 3.11
# Required because: https://github.com/python/cpython/issues/88962
def _remove_kw_only(fn: _FnT) -> _FnT:
  """Remove '_: dataclasses.KW_ONLY' from annotations."""

  @functools.wraps(fn)
  def decorated(cls):
    old_annotations = cls.__annotations__

    if '_' in old_annotations:
      new_annotations = dict(old_annotations)
      new_annotations.pop('_')
    else:
      new_annotations = old_annotations
    try:
      cls.__annotations__ = new_annotations
      return fn(cls)
    finally:
      cls.__annotations__ = old_annotations

  return decorated


@functools.cache
@_remove_kw_only
def _get_type_hints(cls) -> dict[str, _TypeForm]:
  """Wrapper around `typing.get_type_hints` with better error message."""
  try:
    return typing_extensions.get_type_hints(cls, include_extras=True)
  except Exception as e:  # pylint: disable=broad-except
    msg = (
        f'Could not infer typing annotation of {cls.__qualname__} '
        f'defined in {cls.__module__}:\n'
    )
    lines = [f' * {k}: {v!r}' for k, v in cls.__annotations__.items()]
    lines = '\n'.join(lines)

    epy.reraise(e, prefix=msg + lines + '\n')


def _is_annotated_type(hint: _TypeForm) -> bool:
  """`_is_annotated_type(Annotated[int, ...]) == True`."""
  return typing_extensions.get_origin(hint) is Annotated
