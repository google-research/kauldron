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

"""Tools for working with shape specs."""

from kauldron.ktyping import internal_typing
from kauldron.ktyping import scope
from kauldron.ktyping import shape_spec
from kauldron.ktyping import shape_spec_parser


Shape = internal_typing.Shape
CandidateDims = internal_typing.CandidateDims


def shape(spec_str: str, *, nested_ok: bool = False) -> Shape:
  """Evaluates a shape spec in the current scope.

  Args:
    spec_str: The shape spec string to evaluate.
    nested_ok: By default this function will raise an error if the caller is not
      itself a @typechecked function. Disable this check by passing
      nested_ok=True. Use at your own risk.

  Returns:
    The evaluated shape as a tuple of integers.

  Raises:
    ShapeError: If the shape spec cannot be evaluated or is ambiguous under the
    current set of possible dim values. The latter can happen if the
    annotations allow multiple different assignments of dimensions
    (usually because of a Union).
  """
  sscope = scope.get_current_scope(nested_ok=nested_ok, stacklevel=1)
  return _eval_shape(spec_str, sscope.candidates)


def _eval_shape(spec_str: str, candidates: CandidateDims) -> Shape:
  """Evaluates a shape spec for a given set of candidate dim values.

  Args:
    spec_str: The shape spec string to evaluate.
    candidates: The list of candidate dim values to evaluate the shape spec
      against.

  Returns:
    The evaluated shape as a tuple of integers.

  Raises:
    ShapeError: If the shape spec cannot be evaluated or is ambiguous under the
    current set of possible dim values. The latter can happen if the
    annotations allow multiple different assignments of dimensions
    (usually because of a Union).
  """
  spec = shape_spec_parser.parse(spec_str)
  shape_candidates = []
  errors = []
  if not candidates:
    candidates = [{}]  # try without dim values to support fixed shapes

  for dim_values in candidates:
    try:
      shape_candidates.append(spec.evaluate(dim_values))
      errors.append(None)
    except shape_spec.ShapeError as e:
      shape_candidates.append(None)
      errors.append(e)
  valid_shapes = {s for s in shape_candidates if s is not None}
  if not valid_shapes:
    # Can only happen due to errors in each alternative. So we can reraise a
    # ShapeError with the last such error as the cause.
    last_error = [e for e in errors if e is not None][-1]
    exc, cause = shape_spec.ShapeError.from_error(
        f"No possible shape found for {spec_str!r}", last_error
    )
    raise exc from cause

  elif len(valid_shapes) > 1:
    # shape ambiguous under the current set of possible dim_values
    # for each valid shape, get the corresponding dim_values and raise an error
    # TODO(klausg): keep track of the dim value candidates for each shape for
    # the error
    # possibilities = {
    #     s: candidates[shape_candidates.index(s)] for s in valid_shapes
    # }
    raise shape_spec.ShapeError(
        f"{spec_str!r} is ambiguous under the current set of possible"
        " dim_values. Could be one of:/n - "
        + "\n - ".join(f"{k!r}" for k in valid_shapes)
    )
  return valid_shapes.pop()  # return the only valid shape
