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

"""Parser for jaxtyping-like shape specs."""

from __future__ import annotations

import inspect
import sys
import typing

from kauldron.typing import shape_parser
from kauldron.typing import utils


if typing.TYPE_CHECKING:
  Shape = tuple[int, ...]
else:

  class Shape(tuple):
    """Helper to construct concrete shape tuples from shape-specs.

    Example:
    ```
    @typchecked
    def foo(x: Float["*b h w c"], y: Float["h w c"]):
      isinstance(x, Float["h w"])
      print(Shape("h w/2 c+1"))

    foo(np.zeros((7, 32, 32, 3)))  # prints (32, 16, 4)
    ```
    """

    def __new__(cls, spec_str: str) -> tuple[int, ...]:
      _assert_caller_is_typechecked_func()
      spec = parse_shape_spec(spec_str)
      memo = utils.Memo.from_current_context()
      return spec.evaluate(memo)


def _assert_caller_is_typechecked_func() -> None:
  """Raises AssertionError if the calling function is not @typechecked."""
  # First make sure we are not running in a debugger.
  gettrace = getattr(sys, "gettrace", None)
  if gettrace is not None and gettrace():
    return  # likely running in a debugger. Skip the assert since it will fail.

  # The caller function is considered to be the first function in the call stack
  # which is not a list/dict/set-comprehension or generator expression.

  # This function works on the assumption that @typechecked is the innermost
  # decorator of the caller function. Will give false positives otherwise.

  stack = inspect.stack()
  # stack[0].function = "_assert_caller_is_typechecked_func"
  # stack[1].function = either "__new__" (from Shape) or "Dim"
  # ... any number of <listcomp>, <dictcomp>, <setcomp>, and <genexpr>
  # stack[i] = caller function which should be decorated with @typechecked
  # stack[i+1] = "_reraise_with_shape_info"  (if correctly decorated)
  comprehension_names = {"<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"}
  i = 2
  while stack[i].function in comprehension_names and i < len(stack) - 2:
    i += 1

  if stack[i + 1].function != "_reraise_with_shape_info":
    caller_name = stack[i].function
    raise AssertionError(
        "Dim and Shape only work inside of @typechecked functions. But"
        f" {caller_name!r} lacks @typechecked."
    )


def Dim(spec_str: str) -> int:  # pylint: disable=invalid-name
  """Helper to construct concrete Dim (for single-axis Shape)."""
  _assert_caller_is_typechecked_func()
  spec = parse_shape_spec(spec_str)
  memo = utils.Memo.from_current_context()
  ret = spec.evaluate(memo)
  if len(ret) != 1:
    raise utils.ShapeError(
        f"Dim expects a single-axis string, but got : {ret!r}"
    )
  return ret[0]  # pytype: disable=bad-return-type


def parse_shape_spec(spec: str) -> shape_parser.ShapeSpec:
  return shape_parser.parse(spec)
