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

"""Registers custom typeguard checkers for array types."""

import dataclasses
import functools
import operator
import sys
import types
import typing
from typing import Any, Never, NoReturn, Union

from kauldron.ktyping import array_type_meta
from kauldron.ktyping import config
from kauldron.ktyping import errors
from kauldron.ktyping import frame_utils
from kauldron.ktyping import internal_typing
from kauldron.ktyping import pytree
from kauldron.ktyping import scope
from kauldron.ktyping import utils
import typeguard

# forward typeguard functions so that typeguard only needs to be imported here.
TypeCheckError = typeguard.TypeCheckError


def check_type(
    value: object,
    expected_type: Any,
    *,
    nested_ok: bool = False,
) -> Any:
  """Ensure that value matches the expected_type.

  Wraps typeguard.check_type and adds a check that the caller has an active
  ktyping scope.

  Args:
    value: The value to check.
    expected_type: The expected type of the value. Can be any type annotation,
      including basic types(int, str, etc.), container types (list[str],
      dict[str, int], etc.), union types (int | str), and of course ktyping
      array types (Float["b"], UInt8["h w c"]).
    nested_ok: If True, the function will not check if the caller has an active
      ktyping scope.

  Returns:
    The value itself, to allow usage in expression like `x = check_type(...)`.

  Raises:
    AssertionError: If the caller does not have an active ktyping scope and
      `nested_ok` is False.
    typeguard.TypeCheckError: If the value does not match the expected type.
      Note: If the expected_type contains any ktyping types the error will be
      an instance of the ktyping.KTypeCheckError instead.
  """
  if not nested_ok:
    frame_utils.assert_caller_has_active_scope(stacklevel=1)
  return typeguard.check_type(value, expected_type)


def check_type_internal(
    value: object,
    expected_type: Any,
    memo: typeguard.TypeCheckMemo | None = None,
) -> Any:
  """Wraps typeguard.check_type_internal for use without constructing a memo."""
  __traceback_hide__ = True  # pylint: disable=invalid-name,unused-variable
  # Use this instead of typeguard.check_type because the latter does append
  if expected_type is Any:
    return value

  if type(expected_type) is tuple:  # pylint: disable=unidiomatic-typecheck
    expected_type = Union[expected_type]

  if memo is None:
    frame = sys._getframe(1)  # pylint: disable=protected-access
    memo = typeguard.TypeCheckMemo(frame.f_globals, frame.f_locals)
  return typeguard.check_type_internal(value, expected_type, memo)


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


# MARK: MatchResult


@dataclasses.dataclass(kw_only=True, frozen=True)
class MatchResult:
  """Holds the result of a type match check for usage in union checks.

  Attributes:
    value: The value that was matched (e.g. np.array([1, 3, 4])).
    spec: The general type or `ArrayTypeMeta` instance that was matched against
      (e.g. int or Float32["*b n"]).
    type_match: Whether the type of the value matches. (e.g. np.ndarray vs
      tf.Tensor).
    dtype_match: Whether the dtype of the value matches the required dtype (e.g.
      np.float32 vs np.floating).
    shape_match: Whether the shape of the value matches the required shape spec
      (e.g. (2, 4, 5) vs "*b h").
    updated_candidates: The updated set of candidate dim assignments that would
      result from this match.
    error_message: For non-array types this holds the error message from the
      typeguard check. For array types this is None.
  """

  value: Any
  spec: type[Any]
  type_match: bool
  dtype_match: bool
  updated_candidates: internal_typing.CandidateDims
  error_message: str | None = None

  @property
  def shape_match(self) -> bool:
    return bool(self.updated_candidates)

  def __bool__(self) -> bool:
    return self.type_match and self.dtype_match and self.shape_match

  @property
  def should_print_error(self) -> bool:
    """Whether this match failure is interesting enough to print an error for.

    This is used by the array_type_union_checker to filter the errors from a
    complex union match failure. For example: if checking `np.zeros((2, 3, 5))`
    against `TfInt["n"] | TfFloat["a b c"] | NpInt["n"] | NpFloat["a b"]` the
    first match failure (against TfInt["n"]) is skipped because none of its
    type, dtype nor shape matches.
    """
    # Entries that do not match at all are never interesting.
    if not self.type_match and not self.dtype_match and not self.shape_match:
      return False
    # Wrong array type entries are only interesting if they match otherwise.
    if not self.type_match:
      return self.dtype_match and self.shape_match
    # Other failures are considered interesting.
    return True

  @property
  def acceptable_types(self) -> set[type[Any]]:
    """Returns the acceptable types for this match."""
    spec = self.spec  # assign to local variable to avoid pytype error
    if array_type_meta.is_array_type(spec):
      # Unpack the array types e.g. {NpArray, JaxArray} for Float[""]
      return {array_type for array_type in spec.array_types}
    else:
      # For non-array types the spec is the acceptable type itself.
      return {spec}

  @property
  def acceptable_dtypes(self) -> set[str]:
    """Returns the acceptable dtypes for this match."""
    spec = self.spec  # assign to local variable to avoid pytype error
    if array_type_meta.is_array_type(spec):
      return {str(spec.dtype)}
    else:
      return set()

  @property
  def acceptable_shapes(self) -> set[str]:
    """Returns the acceptable shapes for this match."""
    spec = self.spec  # assign to local variable to avoid pytype error
    if array_type_meta.is_array_type(spec):
      # TODO(klausg): partially evaluate the acceptable shapes and list them.
      return {spec.shape_spec}
    else:
      return set()

  def fail_message(self) -> str:
    """Return a message explaining the most salient failure of this match."""
    if self.error_message is not None:
      return self.error_message
    if not self.type_match:
      return errors.array_type_error_message(
          self.value, self.spec.array_types, self.spec
      )
    if not self.dtype_match:
      return errors.dtype_error_message(
          self.value, [self.spec.dtype], self.spec
      )
    if not self.shape_match:
      return errors.shape_error_message(
          self.value, [self.spec.shape_spec], self.spec
      )
    raise ValueError(
        "There is no error message, because"
        f" {utils.format_value(self.value)} correctly matches"
        f" {self.spec}."
    )


def _match_type(value, spec, sscope, memo) -> MatchResult:
  """Check if the value matches the spec and return a TypeMatchResult."""
  if array_type_meta.is_array_type(spec):
    return MatchResult(
        value=value,
        spec=spec,
        type_match=spec.array_types_match(value),
        dtype_match=spec.dtype_matches(value),
        updated_candidates=spec.shape_matches(
            value, sscope.candidates, fstring_locals=sscope.fstring_locals
        ),
    )
  else:  # For non-array type we use the typeguard machinery.
    try:
      typeguard.check_type_internal(value, spec, memo)
    except typeguard.TypeCheckError as exc:
      return MatchResult(
          value=value,
          spec=spec,
          type_match=False,
          dtype_match=False,
          updated_candidates=frozenset(),
          error_message=str(exc),
      )
    return MatchResult(
        value=value,
        spec=spec,
        type_match=True,
        dtype_match=True,  # dtype is not checked for non-array types
        updated_candidates=sscope.candidates,
    )


# MARK: ArrayType chk
def _array_type_checker(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: typeguard.TypeCheckMemo,
) -> None:
  """Custom checker for array types that modifies the memo on success."""
  del args  # unused
  assert array_type_meta.is_array_type(origin_type)
  shape_scope = scope.get_current_scope(nested_ok=True)

  match_result = _match_type(value, origin_type, shape_scope, memo)
  if not match_result:
    raise errors.KTypeCheckError(
        match_result.fail_message(),
        scope=shape_scope,
    )

  shape_scope.candidates = match_result.updated_candidates


# MARK: Union checker
def _array_type_union_checker(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: typeguard.TypeCheckMemo,
) -> None:
  """Custom checker for Union[ArrayType]."""
  del origin_type  # Union or UnionType

  # retrieve active dim value candidates
  sscope = scope.get_current_scope(nested_ok=True)
  # We check all arguments in the union rather than returning early on the
  # first match. We also need to collect the union of all modified candidates.
  # That way we run each check independently of the others and consider each
  # one as a separate candidate set.
  individual_matches: list[MatchResult] = [
      _match_type(value, spec, sscope, memo) for spec in args
  ]

  if any(individual_matches):
    # There is at least one correct match -> no error
    # Update the candidates and return.
    sscope.candidates = functools.reduce(
        frozenset.union, [m.updated_candidates for m in individual_matches if m]
    )
    return

  type_spec = functools.reduce(operator.or_, args)
  # First check if any of the array types matches and raise an error if not.
  if not any(m.type_match for m in individual_matches):
    acceptable_array_types = functools.reduce(
        operator.or_, [m.acceptable_types for m in individual_matches]
    )
    raise errors.KTypeCheckError(
        errors.array_type_error_message(
            value, acceptable_array_types, type_spec=type_spec
        ),
        scope=sscope,
    )

  # Then check if any of the dtypes matches and raise an error if not.
  if not any(m.dtype_match for m in individual_matches):
    acceptable_dtypes = functools.reduce(
        operator.or_, [m.acceptable_dtypes for m in individual_matches]
    )
    raise errors.KTypeCheckError(
        errors.dtype_error_message(
            value, sorted(acceptable_dtypes), type_spec=type_spec
        ),
        scope=sscope,
    )

  # Then check if any of the shapes matches and raise an error if not.
  if not any(m.shape_match for m in individual_matches):
    acceptable_shapes = functools.reduce(
        operator.or_, [m.acceptable_shapes for m in individual_matches]
    )
    raise errors.KTypeCheckError(
        errors.shape_error_message(
            value, acceptable_shapes, type_spec=type_spec
        ),
        scope=sscope,
    )

  # None of the three factors (type, dtype, shape) alone fails, but their
  # combination does.
  # This case is reached if the value does not fully match any of the
  # options in the union, but for each aspect (type, dtype, shape) there is
  # at least one option in the union that would match.
  # For example: `np.float32(0)` vs `Int | Float['b']`.
  # The value matches `Int` on shape (scalar) and `Float` on dtype.
  # But neither matches on both.
  # In this case we compile a list of interesting failures and report them all.
  fail_messages = "\n".join(
      "  - " + m.fail_message()
      for m in individual_matches
      if m.should_print_error
  )
  raise errors.KTypeCheckError(
      "did not match any of its annotations due to a combination"
      f" of:\n{fail_messages}",
      scope=sscope,
  )


# MARK: TypedDict chk
def _typeddict_checker(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: typeguard.TypeCheckMemo,
) -> None:
  """Custom checker for TypedDict that opens a new scope."""
  with scope.create_scope_for(obj=origin_type):
    typeguard._checkers.check_typed_dict(value, origin_type, args, memo)  # pylint: disable=protected-access


# MARK: Dataclass chk
def _dataclass_checker(
    value: Any,
    origin_type: Any,
    args: tuple[Any, ...],
    memo: typeguard.TypeCheckMemo,
) -> None:
  """Custom checker for Dataclass that opens a new scope."""
  del args  # unused
  with scope.create_scope_for(obj=origin_type) as sscope:
    if not isinstance(value, origin_type):
      raise errors.KTypeCheckError(
          f"was of type {type(value)} which is not {origin_type}",
          scope=sscope,
      )
    for field in dataclasses.fields(value):
      field_value = getattr(value, field.name, dataclasses.MISSING)
      try:
        check_type_internal(field_value, field.type, memo)
      except TypeCheckError as exc:
        errors.KTypeCheckError.raise_from_exc(
            exc=exc,
            scope=sscope,
            additional_path_element=f"value of field {field.name!r}",
            maybe_highlight=field.name,
        )


# MARK: PyTree checker


def _pytree_checker(
    value: Any,  # pytree.PyTree[T]
    origin_type: Any,
    args: tuple[Any, ...],
    memo: typeguard.TypeCheckMemo,
) -> None:
  """Custom checker for PyTree."""
  del args  # unused
  import jax  # pylint: disable=g-import-not-at-top

  sscope = scope.get_current_scope(nested_ok=True)
  paths_and_leaves, treedef = jax.tree.flatten_with_path(value)
  del treedef  # TODO(klausg): remember treedef and check structure

  for path, leaf in paths_and_leaves:
    try:
      check_type_internal(leaf, origin_type.leaf_type, memo)
    except TypeCheckError as exc:
      path_str = pytree.jax_path_to_str(path)
      errors.KTypeCheckError.raise_from_exc(
          exc=exc,
          scope=sscope,
          additional_path_element=f"value of tree leaf at path {path_str}",
      )


# MARK: lookup_fn


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
  if scope.is_scope_stack_empty():
    return None

  sscope = scope.get_current_scope(nested_ok=True)
  if not config.get_config(source=sscope.source).typechecking_enabled:
    return None

  # If origin_type is one of the ktyping array types,
  # then return the `_array_type_checker`.
  if array_type_meta.is_array_type(origin_type):
    return _array_type_checker

  # If origin_type is the ktyping PyTree type, then return the `_pytree_checker`
  if pytree.is_pytree_type(origin_type):
    return _pytree_checker

  # If the origin_type is a union which contains at least one array type,
  # then return the `_array_type_union_checker`.
  if origin_type in [Union, types.UnionType]:
    if any(_contains_any_array_type(a) for a in args):
      return _array_type_union_checker

  if typing.is_typeddict(origin_type) and _contains_any_array_type(origin_type):
    return _typeddict_checker

  # Only check dataclasses that are explicitly marked as typechecked.
  if dataclasses.is_dataclass(origin_type) and getattr(
      origin_type, "__ktyping_wrapped__", False
  ):
    return _dataclass_checker
  return None


def _contains_any_array_type(hint: Any) -> bool:
  """Recursively check the typehint and all of its arguments for array types."""
  if array_type_meta.is_array_type(hint):
    return True
  elif typing.is_typeddict(hint):
    annot = utils.get_type_hints(hint)
    return any(_contains_any_array_type(a) for a in annot.values())
  elif dataclasses.is_dataclass(hint):
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
