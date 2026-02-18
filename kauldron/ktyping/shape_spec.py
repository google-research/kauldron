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

"""Parser for jaxtyping-like shape specs."""

from __future__ import annotations

import abc
import dataclasses
import enum
import itertools
import math
import operator
import typing
from typing import Any, Callable, Iterator

from kauldron.ktyping import internal_typing

Shape = internal_typing.Shape
DimValue = internal_typing.DimValue
DimValues = internal_typing.DimValues
UNKNOWN_DIM = internal_typing.UNKNOWN_DIM


class ShapeError(ValueError):
  """Raised when a shape spec cannot be evaluated."""

  @classmethod
  def from_error(cls, msg: str, error: Exception) -> tuple[ShapeError, Any]:
    """Creates a ShapeError with traceback and the cause from another error.

    Usage:
    ```
    try:
      ...
    except Exception as e:
      exc, cause = shape_spec.ShapeError.from_error(
          "Explanatory message.", e
      )
      raise exc from cause
    ```

    This reraise pattern is inspired by
    https://github.com/google/etils/blob/2602f7da504172b6f27cb5d80eaaa4d7ae67eebe/etils/epy/reraise_utils.py#L105C3-L111C61
    Explanation:
     * `with_traceback` will propagate the original stacktrace
     * `from e.__cause__` will:
       * Propagate the original `__cause__` (likely `None`)
       * Set `__suppress_context__` to True, so `__context__` isn't displayed
         This avoid multiple `During handling of the above exception, another
         exception occurred:` messages when nesting rerraises.

    Args:
      msg: The error message to use. The original error message will be
        appended.
      error: The error to extract the message, traceback, and cause from.

    Returns:
      A tuple of the new ShapeError (with traceback from `error`) and the
      original error's cause.
    """
    # Hide the function from the traceback. Is supported by Pytest and IPython 7
    __tracebackhide__ = True  # pylint: disable=unused-variable,invalid-name
    exc = cls(f"{msg} Error: {error!s}").with_traceback(error.__traceback__)
    return exc, error.__cause__


class _Priority(enum.IntEnum):
  OR = enum.auto()
  ADD = enum.auto()
  MUL = enum.auto()
  POW = enum.auto()
  UNARY = enum.auto()
  ATOM = enum.auto()


# ShapeExpression?
# ShapeSpecTree?
@dataclasses.dataclass(frozen=True, init=False)
class ShapeSpec:
  """Top level node of the parsed shape specification."""

  dim_specs: tuple[DimSpec, ...]

  def __init__(self, *dim_specs: DimSpec):
    # Override the default init for convenience.
    # I.e. allowing ShapeSpec(IntDim(42), NamedDims("a", length=1))
    # also use setattr to keep ShapeSpec as a frozen dataclass.
    object.__setattr__(self, "dim_specs", tuple(dim_specs))

  def evaluate(self, dim_values: DimValues) -> Shape:
    """Evaluate this shape spec under the given dim_values."""
    return _concat(s.evaluate(dim_values) for s in self.dim_specs)

  @staticmethod
  def _recursive_match(
      shape: Shape, dim_specs: tuple[DimSpec, ...], dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    if not dim_specs:
      yield shape, dim_values
      return

    match_generator = dim_specs[0].get_all_prefix_matches(shape, dim_values)
    for rest, mod_dim_values in match_generator:
      yield from ShapeSpec._recursive_match(rest, dim_specs[1:], mod_dim_values)

  def match(
      self,
      shape: Shape,
      candidates: internal_typing.CandidateDims,
  ) -> internal_typing.CandidateDims:
    """Try to match the given shape and return set of modified dim values.

    Tries to match the given shape against each candidate in the given
    dim_values. For each match, the dim_values are modified to include
    additional information about unknown dimensions. The returned set of
    modified dim_values is the union of all matches.

    Args:
      shape: The shape to match against.
      candidates: The candidates to consider.

    Returns:
      A set of modified dim_values after matching this ShapeSpec to the
      given shape.
    """
    modified_candidates = set()
    for dim_values in candidates:
      for rest, mod_dim_values in self._recursive_match(
          shape, self.dim_specs, dim_values
      ):
        if not rest and mod_dim_values not in modified_candidates:
          modified_candidates.add(mod_dim_values)
    return frozenset(modified_candidates)

  def __repr__(self):
    return " ".join(repr(ds) for ds in self.dim_specs)


class DimSpec(abc.ABC):
  """Base class for all dimension specs."""

  @abc.abstractmethod
  def evaluate(self, dim_values: DimValues) -> Shape:
    """Evaluate this dimension spec under the given dim_values.

    Args:
      dim_values: The dim values to evaluate against.

    Returns: The evaluated dimension spec as a tuple of int dimension values.

    Raises:
      ShapeError: If the DimSpec cannot be evaluated / is underconstraint.
    """

  @abc.abstractmethod
  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    """Iterate all possible evals of this DimSpec under the given dim_values.

    Args:
      dim_values: The dim values to evaluate against.

    Returns: An iterator that for each possible evaluation yields
      the evaluated dimension spec.

    Raises:
      ShapeError: If the DimSpec cannot be evaluated / is underconstraint.
    """

  @abc.abstractmethod
  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    """Returns all possible prefix matches of this DimSpec for a given shape.

    Most DimSpecs will only have a single prefix match, but some DimSpecs
    (e.g. NamedDims with length=None) can have multiple.

    Args:
      shape: The shape values to match against.
      dim_values: The active dim_values.

    Returns:
      An iterator that for each possible match yields a tuple of the remaining
      unmatched shape values (suffix) and the associated (possibly modified)
      dim_values.
    """

  @property
  def priority(self) -> int:
    """Only used for omitting parentheses in __repr__."""
    return _Priority.ATOM


@dataclasses.dataclass
class IntDim(DimSpec):
  """Single dimension with a fixed value like 128 or 5."""

  value: int
  broadcastable: bool = False

  def evaluate(self, dim_values: DimValues) -> Shape:
    if self.broadcastable:
      raise ShapeError(f"Cannot evaluate a broadcastable dim: {self!r}")
    return (self.value,)

  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    if self.broadcastable:
      yield (1,)
    yield (self.value,)

  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    if not shape:
      return

    elif shape[0] == self.value:
      # first dim matches
      yield shape[1:], dim_values
    elif self.broadcastable and shape[0] == 1:
      # first dim is broadcastable
      yield shape[1:], dim_values

  def __repr__(self):
    prefix = "#" if self.broadcastable else ""
    return f"{prefix}{self.value}"


def _unknown_to_one(shape: DimValue) -> Shape:
  shape = tuple(1 if d == UNKNOWN_DIM else d for d in shape)
  return typing.cast(Shape, shape)


def _one_to_unknown(shape: Shape) -> DimValue:
  return tuple(UNKNOWN_DIM if d == 1 else d for d in shape)


def _consistent_with(
    shape: Shape, dim_values: DimValue, broadcastable: bool
) -> bool:
  if len(shape) != len(dim_values):
    return False

  return all(
      s == c or c == UNKNOWN_DIM or (s == 1 and broadcastable)
      for s, c in zip(shape, dim_values)
  )


@dataclasses.dataclass(frozen=True)
class NamedDims(DimSpec):
  """Represents a (non-anonymous) dimension with a name."""

  name: str

  _: dataclasses.KW_ONLY

  length: None | int | tuple[int, int | None] = 1

  broadcastable: bool = False

  @property
  def min_length(self) -> int:
    if self.length is None:
      return 0
    elif isinstance(self.length, int):
      return self.length
    else:
      # tuple[min, max]
      return self.length[0]

  @property
  def max_length(self) -> int | None:
    if self.length is None:
      return None
    elif isinstance(self.length, int):
      return self.length
    else:
      # tuple[min, max]
      return self.length[1]

  def evaluate(self, dim_values: DimValues) -> tuple[int, ...]:
    if self.broadcastable:
      raise ShapeError(f"Cannot evaluate a broadcastable dimension: {self!r}")

    dim_value = dim_values.get(self.name, None)
    if dim_value is None:
      raise ShapeError(
          f"No value known for {self!r}. "
          f"Known values are: {sorted(dim_values.keys())}"
      )
    # verify that the active dim_values match the min/max length of this spec
    if self.min_length <= len(dim_value) and (
        self.max_length is None or len(dim_value) <= self.max_length
    ):
      return _unknown_to_one(dim_value)

    raise ShapeError(
        f"The active dim_value {self!r}={dim_value} is not "
        "consistent with its length spec. It should have between "
        f"{self.min_length} and {self.max_length} dimensions."
    )

  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    dim_value = dim_values.get(self.name, None)
    if dim_value is None:
      return  # the dimension is underconstrained.

    # verify that the active dim_values match the min/max length of this spec
    if self.min_length > len(dim_value):
      return  # too short

    if self.max_length is not None and len(dim_value) > self.max_length:
      return  # too long

    shape = _unknown_to_one(dim_value)
    if not self.broadcastable:
      yield shape
    else:
      # if the dim is broadcastable each individual value could be 1 instead
      # e.g. (8, 5) could  (8, 5), (8, 1), (1, 5), (1, 1)
      yield from set(itertools.product(*[(x, 1) for x in shape]))

  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    dim_value = dim_values.get(self.name, None)
    if dim_value is not None:
      prefix, rest = shape[: len(dim_value)], shape[len(dim_value) :]
      # check if the dim_value is consistent with a prefix of the given dims
      if _consistent_with(prefix, dim_value, self.broadcastable):
        # we might need to adjust the dim_value to add additional information
        # about unknown dimensions
        if self.broadcastable:
          # We might gain new information about shape values that are not 1.
          new_dim_value = tuple(
              shape_dim if shape_dim != 1 else stored_dim
              for shape_dim, stored_dim in zip(shape, dim_value)
          )

          yield rest, dim_values | {self.name: new_dim_value}
        else:
          # replace the dim_value with the prefix because it is fully specified
          yield rest, dim_values | {self.name: prefix}
      return
    else:
      # if the dim_value is not known, we need to try all prefixes
      max_length = len(shape) if self.max_length is None else self.max_length
      for i in range(self.min_length, max_length + 1):
        prefix, rest = shape[:i], shape[i:]
        if self.broadcastable:
          prefix = _one_to_unknown(prefix)
        if self.min_length <= len(prefix) <= max_length:
          yield rest, dim_values | {self.name: prefix}

  def __repr__(self):
    star = "*" if self.min_length == 0 and self.max_length is None else ""
    plus = "+" if self.min_length == 1 and self.max_length is None else ""
    broadcast = "#" if self.broadcastable else ""
    return f"{star}{plus}{broadcast}{self.name}"


@dataclasses.dataclass(frozen=True)
class AnonDims(DimSpec):
  """Represents an anonymous dimension."""

  name: str | None = None

  _: dataclasses.KW_ONLY

  length: None | int | tuple[int, int | None] = 1

  @property
  def min_length(self) -> int:
    if self.length is None:
      return 0
    elif isinstance(self.length, int):
      return self.length
    else:
      # tuple[min, max]
      return self.length[0]

  @property
  def max_length(self) -> int | None:
    if self.length is None:
      return None
    elif isinstance(self.length, int):
      return self.length
    else:
      # tuple[min, max]
      return self.length[1]

  def evaluate(self, dim_values: DimValues) -> tuple[int, ...]:
    raise ShapeError(f"Cannot evaluate anonymous dimension: {self!r}")

  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    yield from ()  # cannot be evaluated / is underconstrained

  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    max_length = len(shape) if self.max_length is None else self.max_length
    for i in range(self.min_length, max_length + 1):
      prefix, rest = shape[:i], shape[i:]
      if self.min_length <= len(prefix) <= max_length:
        yield rest, dim_values

  def __repr__(self):

    star = "*" if self.min_length == 0 and self.max_length is None else ""
    plus = "+" if self.min_length == 1 and self.max_length is None else ""
    name = "" if self.name is None else self.name
    if star and not self.name:
      return "..."
    else:
      return f"{star}{plus}_{name}"


@dataclasses.dataclass(frozen=True)
class OptionalDim(DimSpec):
  """A dimension that might be present or not."""

  child: DimSpec

  def evaluate(self, dim_values: DimValues) -> Shape:
    raise ShapeError(f"Cannot evaluate optional dimensions ('?'): {self!r}")

  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    yield ()
    yield from self.child.evaluate_all(dim_values)

  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    yield shape, dim_values  # ignore the optional dim
    yield from self.child.get_all_prefix_matches(shape, dim_values)

  @property
  def priority(self) -> int:
    return _Priority.UNARY

  def __repr__(self):
    if self.child.priority < self.priority:
      return f"({self.child!r})?"
    else:
      return f"{self.child!r}?"


BinOp = Callable[[Any, Any], Any]


@dataclasses.dataclass
class Operator:
  symbol: str
  fn: BinOp
  priority: _Priority


OPERATORS = [
    Operator("+", operator.add, _Priority.ADD),
    Operator("-", operator.sub, _Priority.ADD),
    Operator("*", operator.mul, _Priority.MUL),
    Operator("/", operator.truediv, _Priority.MUL),
    Operator("//", operator.floordiv, _Priority.MUL),
    Operator("%", operator.mod, _Priority.MUL),
    Operator("**", operator.pow, _Priority.POW),
]

SYMBOL_2_OPERATOR = {o.symbol: o for o in OPERATORS}


def _concat(shapes: Iterator[Shape] | tuple[Shape, ...]) -> Shape:
  """Concatenate a sequence of shapes into a single shape."""
  if isinstance(shapes, Iterator):
    return tuple(itertools.chain.from_iterable(shapes))
  else:
    return tuple(itertools.chain(*shapes))


@dataclasses.dataclass
class FunctionDim(DimSpec):
  """Function based dimension specs like "min(a,b)" or "sum(*batch)."""

  name: str
  fn: Callable[..., int]
  arguments: list[DimSpec]

  def evaluate(self, dim_values: DimValues) -> Shape:
    vals = _concat(arg.evaluate(dim_values) for arg in self.arguments)
    return (self.fn(vals),)

  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    arg_generators = [a.evaluate_all(dim_values) for a in self.arguments]
    for s in itertools.product(*arg_generators):
      yield (self.fn(_concat(s)),)

  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    # TODO(klausg): Could be used to infer unknown dims.
    for eval_shape in self.evaluate_all(dim_values):
      if eval_shape[0] == shape[0]:
        yield shape[1:], dim_values

  def __repr__(self):
    arg_list = ",".join(repr(a) for a in self.arguments)
    return f"{self.name}({arg_list})"


NAME_2_FUNC = {"sum": sum, "min": min, "max": max, "prod": math.prod}


@dataclasses.dataclass
class BinaryOpDim(DimSpec):
  """Binary ops for dim specs such as "H*W" or "C+1"."""

  op: Operator
  left: DimSpec
  right: DimSpec

  def evaluate(self, dim_values: DimValues) -> Shape:
    # TODO(klausg): explicit error if left/right is not a single dim
    (left,) = self.left.evaluate(dim_values)  # unpack (has to be 1-dim)
    (right,) = self.right.evaluate(dim_values)  # unpack (has to be 1-dim)
    return (self.op.fn(left, right),)

  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    for shape_l in self.left.evaluate_all(dim_values):
      for shape_r in self.right.evaluate_all(dim_values):
        if len(shape_l) != 1 or len(shape_r) != 1:
          continue  # only support binary ops on single dims
        yield (self.op.fn(shape_l[0], shape_r[0]),)

  @property
  def priority(self) -> int:
    return self.op.priority

  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    left_shapes = self.left.evaluate_all(dim_values)
    right_shapes = self.right.evaluate_all(dim_values)
    # Filter out multi-dim because we only support binary ops on single dims.
    left_shapes = [s[0] for s in left_shapes if len(s) == 1]
    right_shapes = [s[0] for s in right_shapes if len(s) == 1]
    # Check all possible combinations of left and right shapes for a match.
    for left, right in itertools.product(left_shapes, right_shapes):
      dim_value = self.op.fn(left, right)
      if dim_value == shape[0]:
        yield shape[1:], dim_values

    # If the right side is underconstrained, we can try to infer it.
    if left_shapes and not right_shapes:
      for left in left_shapes:
        rshape = self.solve_right(left=left, target=shape[0])
        if rshape is None:
          continue
        yield from self.right.get_all_prefix_matches(
            (rshape,) + shape[1:], dim_values
        )
    # If the left side is underconstrained, we can try to infer it.
    if not left_shapes and right_shapes:
      for right in right_shapes:
        lshape = self.solve_left(right=right, target=shape[0])
        if lshape is None:
          continue
        yield from self.left.get_all_prefix_matches(
            (lshape,) + shape[1:], dim_values
        )

  def solve_right(self, *, left: int, target: int) -> int | None:
    """Solves the right shape given the left shape and the target value."""
    match self.op.symbol:
      case "+":  # target = left + right  ==>  right = target - left
        return target - left
      case "-":  # target = left - right  ==>  right = left - target
        return left - target
      case "*":  # target = left * right  ==>  right = target // left
        right = target // left
        return right if left * right == target else None
      case "//":  # target = left // right  ==>  right = left // target
        right = left // target
        return right if left // right == target else None
      case "/":  # target = left / right  ==>  right = left / target
        right = left // target
        return right if left / right == target else None
      case "%":
        return None  # Too many solutions to this equation.
      case _:
        raise ValueError(f"Unsupported operator: {self.op.symbol}")

  def solve_left(self, *, right: int, target: int) -> int | None:
    """Solves the left shape given the right shape and the target value."""
    match self.op.symbol:
      case "+":  # target = left + right  ==>  left = target - right
        return target - right
      case "-":  # target = left - right  ==>  left = target + right
        return target + right
      case "*":  # target = left * right  ==>  left = target // right
        left = target // right
        return left if left * right == target else None
      case "//":  # target = left // right  ==>  left = target * right
        return target * right
      case "/":  # target = left / right  ==>  left = target * right
        return target * right
      case "%":
        return None  # Too many solutions to this equation.
      case _:
        raise ValueError(f"Unsupported operator: {self.op.symbol}")

  def __repr__(self):
    left_repr = (
        repr(self.left)
        if self.priority < self.left.priority
        else f"({self.left!r})"
    )
    right_repr = (
        repr(self.right)
        if self.priority < self.right.priority
        else f"({self.right!r})"
    )
    return f"{left_repr}{self.op.symbol}{right_repr}"


@dataclasses.dataclass
class ChoiceDim(DimSpec):
  """Represent a dimension that can be either of two other dimension specs."""

  left: DimSpec
  right: DimSpec

  def evaluate(self, dim_values: DimValues) -> Shape:
    raise ShapeError(
        "Cannot evaluate dimensions that contain an OR operator ('|'):"
        f" {self!r}"
    )

  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    yield from self.left.evaluate_all(dim_values)
    yield from self.right.evaluate_all(dim_values)

  @property
  def priority(self) -> int:
    return _Priority.OR

  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    yield from self.left.get_all_prefix_matches(shape, dim_values)
    yield from self.right.get_all_prefix_matches(shape, dim_values)

  def __repr__(self):
    left_repr = (
        repr(self.left)
        if self.priority < self.left.priority
        else f"({self.left!r})"
    )
    right_repr = (
        repr(self.right)
        if self.priority < self.right.priority
        else f"({self.right!r})"
    )
    return f"{left_repr}|{right_repr}"


@dataclasses.dataclass
class NegatedDim(DimSpec):
  """Negation of a dim spec, e.g. "-h"."""

  child: DimSpec

  def evaluate(self, dim_values: DimValues) -> Shape:
    dim_value = self.child.evaluate(dim_values)
    assert len(dim_value) == 1, "negation of multi-dim should never happen"
    return (-dim_value[0],)

  def evaluate_all(self, dim_values: DimValues) -> Iterator[Shape]:
    for shape in self.child.evaluate_all(dim_values):
      if len(shape) != 1:
        continue  # only support negation of single dims
      yield (-shape[0],)

  @property
  def priority(self) -> int:
    return _Priority.UNARY

  def get_all_prefix_matches(
      self, shape: Shape, dim_values: DimValues
  ) -> Iterator[tuple[Shape, DimValues]]:
    prefix, rest = shape[:1], shape[1:]
    neg_prefix = (-prefix[0],)

    yield from self.child.get_all_prefix_matches(neg_prefix + rest, dim_values)

  def __repr__(self):
    if self.priority < self.child.priority:
      return f"-{self.child!r}"
    else:
      return f"-({self.child!r})"
