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

from kauldron.ktyping import errors
from kauldron.ktyping import typeguard_checkers  # pylint: disable=unused-import
from kauldron.ktyping.array_types import Float, Int  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.ktyping.decorator import typechecked  # pylint: disable=g-importing-member
import numpy as np
import pytest
import tensorflow as tf


@typechecked
def custom_function(x: Float["a b"], y: Int["b c"]) -> Float["b"]:
  del y
  return x[0]


def test_array_type_check():

  x = np.zeros((2, 3), dtype=np.float32)
  y = np.zeros((3, 5), dtype=np.int32)
  assert custom_function(x, y).shape == (3,)

  with pytest.raises(errors.KTypeCheckError) as exc_info:
    custom_function(x, tf.constant(y))

  exc = exc_info.value
  msg = str(exc)

  assert msg.startswith(
      "argument y = tf.i32[3 5] was of type"
      " tensorflow.python.framework.ops.EagerTensor"
      " which is not an instance of JaxArray|NpArray (required by Int['b c'])"
  )

  assert exc.origin_block.startswith(
      f"Origin: function 'custom_function' at {__file__}:"
  )
  assert exc.origin_block in msg

  assert exc.arguments_block == """\
Arguments:
  x: Float['a b'] = np.f32[2 3]
> y: Int['b c'] = tf.i32[3 5]"""
  assert exc.arguments_block in msg

  assert exc.return_block == "Return: Float['b']"
  assert exc.return_block in msg

  assert exc.candidates_block == "Dim Assignments:\n - {a: 2, b: 3}"
  assert exc.candidates_block in msg


def test_format_dim_assignment_with_unknown_dim():
  """UNKNOWN_DIM should be formatted as '#' in dim assignments."""
  from kauldron.ktyping import internal_typing  # pylint: disable=g-import-not-at-top

  unknown = internal_typing.UNKNOWN_DIM

  # Single unknown dim
  result = errors._format_dim_assignment("batch", (unknown,))
  assert result == "batch: #", result

  # Multi-dim with unknown
  result = errors._format_dim_assignment("data", (6, unknown))
  assert result == "*data: (6, #)", result

  # Multi-dim all unknown
  result = errors._format_dim_assignment("b", (unknown, unknown))
  assert result == "*b: (#, #)", result

  # No unknown (sanity check)
  result = errors._format_dim_assignment("x", (3, 5))
  assert result == "*x: (3, 5)", result


@typechecked
def _broadcastable_fn(
    x: Float["*#batch n"],
    y: Int["*#batch m"],
) -> Float["n"]:
  del y
  return x[..., :, 0]  # wrong return shape to trigger error


def test_unknown_dim_in_error_message():
  """UNKNOWN_DIM in candidates_block should show '#' not the enum repr."""
  x = np.zeros((1, 6, 4), dtype=np.float32)  # #*batch=(1, 6) -> (#, 6)
  y = np.zeros((1, 6, 3), dtype=np.int32)

  with pytest.raises(errors.KTypeCheckError) as exc_info:
    _broadcastable_fn(x, y)

  msg = str(exc_info.value)
  # The ugly enum repr should NOT appear in the error message
  assert (
      "UnknownDim" not in msg
  ), f"UnknownDim leaked into error message:\n{msg}"
  assert (
      "UNKNOWN_DIM" not in msg
  ), f"UNKNOWN_DIM leaked into error message:\n{msg}"
  # The '#' formatting should be used instead
  assert "#" in exc_info.value.candidates_block
