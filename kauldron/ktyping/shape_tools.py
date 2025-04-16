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

"""Tools for working with shape specs."""

from kauldron.ktyping import scope
from kauldron.ktyping import shape_spec
from kauldron.ktyping import shape_spec_parser
from kauldron.ktyping.constraints import ConstraintAlternatives  # pylint: disable=g-importing-member
from kauldron.ktyping.internal_typing import Shape  # pylint: disable=g-importing-member


def shape(spec_str: str, subscope_ok: bool = False) -> Shape:
  """Evaluates a shape spec in the current scope.

  Args:
    spec_str: The shape spec string to evaluate.
    subscope_ok: By default this function will raise an error if the caller is
      not itself a @typechecked function. Disable this check by passing
      subscope_ok=True. Use at your own risk.

  Returns:
    The evaluated shape as a tuple of integers.

  Raises:
    ShapeError: If the shape spec cannot be evaluated or is ambiguous under the
    current set of possible constraints. The latter can happen if the
    annotations allow multiple different assignments of dimensions
    (usually because of a Union).
  """
  if not subscope_ok:
    scope.assert_caller_has_active_scope()
  dims = scope.get_current_scope(subscope_ok=True)
  return eval_shape(spec_str, dims.alternatives)


def dim(spec_str: str, subscope_ok: bool = False) -> int:
  """Evaluates a shape spec in the current scope.

  Args:
    spec_str: The shape spec string to evaluate.
    subscope_ok: By default this function will raise an error if the caller is
      not itself a @typechecked function. Disable this check by passing
      subscope_ok=True. Use at your own risk.

  Returns:
    The size of the evaluated dimension as an int.

  Raises:
    ShapeError: If the shape spec cannot be evaluated or is ambiguous under the
    current set of possible constraints. The latter can happen if the
    annotations allow multiple different assignments of dimensions
    (usually because of a Union).
  """
  if not subscope_ok:
    scope.assert_caller_has_active_scope()
  alternatives = scope.get_current_scope(subscope_ok=True).alternatives
  shape_ = eval_shape(spec_str, alternatives)
  if len(shape_) != 1:
    raise TypeError(
        f"Dim expects a single-axis string, but got: {spec_str!r} = {shape_!r}"
    )
  return shape_[0]  # pytype: disable=bad-return-type


def eval_shape(spec_str: str, alternatives: ConstraintAlternatives) -> Shape:
  """Evaluates a shape spec for a given set of alternative constraints.

  Args:
    spec_str: The shape spec string to evaluate.
    alternatives: The list of alternative constraints to evaluate the shape spec
      against.

  Returns:
    The evaluated shape as a tuple of integers.

  Raises:
    ShapeError: If the shape spec cannot be evaluated or is ambiguous under the
    current set of possible constraints. The latter can happen if the
    annotations allow multiple different assignments of dimensions
    (usually because of a Union).
  """
  spec = shape_spec_parser.parse(spec_str)
  shape_alternatives = []
  errors = []
  if not alternatives:
    alternatives = [{}]  # try without constraints to support fixed shapes

  for constraint in alternatives:
    try:
      shape_alternatives.append(spec.evaluate(constraint))
      errors.append(None)
    except shape_spec.ShapeError as e:
      shape_alternatives.append(None)
      errors.append(e)
  valid_shapes = {s for s in shape_alternatives if s is not None}
  if not valid_shapes:
    # Can only happen due to errors in each alternative. So we can reraise a
    # ShapeError with the last such error as the cause.
    # TODO(klausg): could also try to list all errors
    last_error = [e for e in errors if e is not None][-1]
    exc, cause = shape_spec.ShapeError.from_error(
        f"No possible shape found for {spec_str!r}", last_error
    )
    raise exc from cause

  elif len(valid_shapes) > 1:
    # shape ambiguous under the current set of possible constraints
    # for each valid shape, get the corresponding constraint and raise an error
    # TODO(klausg): keep track of the constraint for each shape for the error
    # possibilities = {
    #     s: alternatives[shape_alternatives.index(s)] for s in valid_shapes
    # }
    raise shape_spec.ShapeError(
        f"{spec_str!r} is ambiguous under the current set of possible"
        " constraints. Could be one of:/n - "
        + "\n - ".join(f"{k!r}" for k in valid_shapes)
    )
  return valid_shapes.pop()  # return the only valid shape
