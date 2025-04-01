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

"""Registers custom typeguard checkers for array types."""

import types
import typing
from typing import Any, Never, NoReturn, Union
from kauldron.ktyping import array_types
from kauldron.ktyping import scope
from kauldron.ktyping import utils
import typeguard

# forward typeguard functions so that typeguard only needs to be imported here.
check_type = typeguard.check_type
check_type_internal = typeguard.check_type_internal
TypeCheckError = typeguard.TypeCheckError


def assert_not_never(fn, annot: Any) -> None:
  """Raises a TypeCheckError if the annotation is Never."""
  if annot is Never:
    raise typeguard.TypeCheckError(
        f"Function {fn.__name__} declared never to be called but it was."
    )


def assert_not_noreturn(fn, annot: Any) -> None:
  """Raises a TypeCheckError if the annotation is NoReturn or Never."""
  if annot in (NoReturn, Never):
    raise typeguard.TypeCheckError(
        f"Function {fn.__name__} declared never to return but it did."
    )


def _array_types_checker_lookup(
    origin_type: Any, args: tuple[Any, ...], extras: tuple[Any, ...]
) -> typeguard.TypeCheckerCallable | None:
  """Custom lookup function for array types."""
  del extras
  # Disable all custom checkers if there is no active scope.
  # This is to avoid changing the behavior of typeguard when not using ktyping.
  # Note: this only partially works, because it will still change the behavior
  # of typeguard for functions called from inside an active ktyping scope.
  # Because of that we also make sure to only return a custom checker if the
  # typehint contains an array type of any kind.
  if not scope.has_active_scope():
    return None

  # If origin_type is one of the ktyping array types,
  # then return the `_array_type_checker`.
  if array_types.is_array_type(origin_type):
    return _array_type_checker
  # If the origin_type is a union which contains at least one array type,
  # then return the `_array_type_union_checker`.
  if origin_type in [Union, types.UnionType]:
    if any(_contains_any_array_type(a) for a in args):
      return _array_type_union_checker

  if typing.is_typeddict(origin_type) and _contains_any_array_type(origin_type):
    return _typeddict_checker
  # TODO(klausg): custom checker for dataclass
  return None


def _array_type_checker(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: typeguard.TypeCheckMemo,
) -> None:
  """Custom checker for array types that modifies the memo on success."""
  del args, memo  # unused
  assert array_types.is_array_type(origin_type)
  shape_scope = scope.get_current_scope(subscope_ok=True)
  # TODO(klausg): add additional information to the exception
  # Check if the value is an instance of any of the array types.
  if not origin_type.array_types_match(value):
    array_types_str = " | ".join(
        utils.get_type_name(a) for a in origin_type.array_types
    )
    raise typeguard.TypeCheckError(
        f" ({type(value)}) is not an instance of {array_types_str} (as"
        f" required by {origin_type!r})."
    )
  # Check if the value has the required dtype.
  if not origin_type.dtype_matches(value):
    raise typeguard.TypeCheckError(
        f" ({type(value)}) has dtype {utils.get_dtype_str(value)} which"
        f" is not compatible with {origin_type!r}."
    )
  # Finally check if the shape of the value matches the shape spec.
  updated_alternatives = origin_type.shape_matches(
      value, shape_scope.alternatives, bound_args=shape_scope.bound_args  # pylint: disable=protected-access
  )
  if not updated_alternatives:
    raise typeguard.TypeCheckError(
        f" ({type(value)}) has shape {value.shape} which is not compatible"
        f" with {origin_type!r}."
    )
  shape_scope.alternatives = updated_alternatives


def _array_type_union_checker(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: typeguard.TypeCheckMemo,
) -> None:
  """Custom checker for Union[ArrayType]."""
  del origin_type  # Union or UnionType
  # retrieve active constraint alternatives
  dims = scope.get_current_scope(subscope_ok=True)
  # We check all arguments in the union rather than returning early on the
  # first match. We also need to reset the alternatives after each check, and
  # collect the union of all modified alternatives.
  # That way we run each check independently of the others and consider each
  # one as a separate alternative.
  original_alternatives = dims.alternatives
  new_alternatives = set()
  for arg in args:
    typeguard.check_type_internal(value, arg, memo)
    new_alternatives.update(dims.alternatives)
    dims.alternatives = original_alternatives
  dims.alternatives = new_alternatives


def _typeddict_checker(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: typeguard.TypeCheckMemo,
) -> None:
  """Custom checker for TypedDict that opens a new scope."""
  with scope.ShapeScope():
    typeguard._checkers.check_typed_dict(value, origin_type, args, memo)  # pylint: disable=protected-access


def _contains_any_array_type(hint: Any) -> bool:
  """Recursively check the typehint and all of its arguments for array types."""
  if array_types.is_array_type(hint):
    return True
  if typing.is_typeddict(hint):
    annot = utils.get_type_hints(hint)
    return any(_contains_any_array_type(a) for a in annot.values())
  args = typing.get_args(hint)
  if args is not None:
    return any(_contains_any_array_type(a) for a in args)
  return False


def add_custom_checker_lookup_fn(
    lookup_fn: typeguard.TypeCheckLookupCallback,
) -> None:
  """Add custom lookup function to typeguard or replace existing one.

  Checks not for equality but for qualname, to avoid many copies when
  reloading modules from colab.

  Args:
    lookup_fn: The lookup function to add or replace. It must be a callable with
      the signature of a typeguard.TypeCheckLookupCallback.
  """
  # Check not for equality but for qualname, to avoid many copies when
  # reloading modules from colab
  checker_lookup_fns = typeguard.checker_lookup_functions
  for i, f in enumerate(checker_lookup_fns):
    if getattr(f, "__qualname__", None) == getattr(
        lookup_fn, "__qualname__", None
    ):
      # replace
      checker_lookup_fns[i : i + 1] = [lookup_fn]
      break
  else:  # prepend
    checker_lookup_fns[:0] = [lookup_fn]


add_custom_checker_lookup_fn(_array_types_checker_lookup)
