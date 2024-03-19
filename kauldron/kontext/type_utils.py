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

"""Typing annotation utils."""

from __future__ import annotations

import functools
import typing
from typing import Annotated, Any, TypeVar, Union

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


def _is_optional(annotation: type[Any]) -> bool:
  """Check if a typing annotation is Optional[...]."""
  origin = typing.get_origin(annotation)
  args = typing.get_args(annotation)
  return origin is Union and type(None) in args


def _is_hint_annotated_with(hint: _TypeForm, annotated_token: _Token) -> bool:
  """Returns `True` if the hint is annotated with `annotated_token`."""
  origin = typing.get_origin(hint)

  if origin is None:  # Leaf (non-generic)
    return False

  # For generic types, check if the type itself is Annotated
  if origin == typing.Annotated and any(
      a is annotated_token for a in hint.__metadata__
  ):
    return True

  # Recurse
  # Note that `get_args` can return non-type values (e.g. `...`,
  # `[]` for Callable, and any values for `Annotated`)
  # `Callable[[int], Any]` won't be recursed into.
  return any(
      _is_hint_annotated_with(arg, annotated_token)
      for arg in typing.get_args(hint)
  )


@functools.cache
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
