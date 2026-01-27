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

"""Utilities for interacting with stack frames."""

import inspect
import os
import sys

# Local variable that is set to mark a frame as having an active scope.
# Used by assert_caller_has_active_scope()
_SCOPE_COUNTER = "__ktyping_scope_counter__"

# Local variable that is set to mark a frame as ignored by get_caller_frame()
# Used to ignore internal wrappers and decorators.
_SCOPE_IGNORE_MARKER = "__ktyping_ignore_frame__"

_LAMBDA_AND_COMPREHENSIONS = frozenset((
    "<lambda>",
    "<listcomp>",
    "<dictcomp>",
    "<setcomp>",
    "<genexpr>",
))


# MARK: NoActiveScope
class NoActiveScopeError(RuntimeError):
  """Raised when there is no active scope."""

  def __init__(self, frame: inspect.FrameInfo | None = None):
    stack_summary = _get_stack_summary(limit=10, frame_to_highlight=frame)
    super().__init__(
        "No active scope found. Has to be called within the context of a"
        " @typechecked decorator or context manager. Ten topmost frames:"
        f"\n{stack_summary}"
    )


# MARK: get_caller_frame
def get_caller_frame(
    stacklevel: int = 0,
) -> inspect.FrameInfo:
  """Returns FrameInfo for the caller function while ignoring lambdas etc.

  Args:
    stacklevel: The number of (non-ignored) stack frames to skip. This is useful
      for ignoring wrappers and decorators. stacklevel=0 returns the frame of
      the function calling get_caller_frame().
  """
  __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

  filtered_stack = (f for f in inspect.stack() if not _should_ignore(f))
  for i, frame in enumerate(filtered_stack):
    if i == stacklevel:
      return frame
  raise ValueError(f"{stacklevel=} is out of bounds.")


def _should_ignore(frame: inspect.FrameInfo) -> bool:
  return frame.function in _LAMBDA_AND_COMPREHENSIONS or bool(
      frame.frame.f_locals.get(_SCOPE_IGNORE_MARKER, False)
  )


# MARK: mark/unmark
def mark_frame_as_active_scope(frame: inspect.FrameInfo) -> None:
  """Adds/increments a SCOPE_COUNTER marker in the frame's local variables."""
  active_scope_count = frame.frame.f_locals.get(_SCOPE_COUNTER, 0)
  frame.frame.f_locals[_SCOPE_COUNTER] = active_scope_count + 1


def unmark_frame_as_active_scope(frame: inspect.FrameInfo) -> None:
  """Removes/decrements SCOPE_COUNTER marker in the frame's local variables."""
  active_scope_count = frame.frame.f_locals.get(_SCOPE_COUNTER, 0)
  if not active_scope_count and not is_running_in_debugger():
    raise AssertionError(f"Not an active DimScope: {frame.function}")
  frame.frame.f_locals[_SCOPE_COUNTER] = active_scope_count - 1


# MARK: assert active
def assert_caller_has_active_scope(stacklevel: int = 0) -> None:
  """Raises AssertionError if the calling function is not @typechecked."""
  __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

  # Make sure we are not running in a debugger, because this check would break.
  if is_running_in_debugger():
    return

  frame = get_caller_frame(stacklevel=stacklevel)

  if not has_active_scope(frame):
    raise NoActiveScopeError(frame=frame)


def has_active_scope(frame: inspect.FrameInfo) -> bool:
  """Returns True if the given frame has an active scope."""

  is_active_scope = frame.frame.f_locals.get(_SCOPE_COUNTER, 0)
  if is_active_scope:
    return True

  # check if the parent frame is a wrapper (which always has an active scope)
  parent_frame = frame.frame.f_back
  if parent_frame is None:
    return False

  return bool(parent_frame.f_locals.get("__ktyping_wrapper__", False))


# MARK: is_running_in_debugger
def is_running_in_debugger() -> bool:
  """Returns True if running in a debugger."""
  # gettrace is set by the debugger but also by coverage mode in tests.
  gettrace = getattr(sys, "gettrace", lambda: False)
  in_coverage_mode = os.environ.get("COVERAGE_RUN", False)
  return gettrace() and not in_coverage_mode


def _get_stack_summary(
    limit: int | None = None,
    frame_to_highlight: inspect.FrameInfo | None = None,
) -> str:
  """Returns a text summary of the stack.

  Args:
    limit: The maximum number of frames to print. If None, print all frames.
    frame_to_highlight: The frame to highlight with >>. Optional.

  Returns:
    A string with the stack summary.

  Example output:
        _get_stack_summary: ---
        __init__: ---
        assert_caller_has_active_scope: [ignored]
    >> fn: ---
        test_assert_caller_has_active_scope: ACTIVE SCOPE
        pytest_pyfunc_call: ---
  """

  def _format_frame_line(frame: inspect.FrameInfo) -> str:
    is_active_scope = frame.frame.f_locals.get(_SCOPE_COUNTER, 0)

    indent = ">>" if frame == frame_to_highlight else "  "
    if _should_ignore(frame):
      description = "[ignored]"
    elif is_active_scope:
      description = "ACTIVE SCOPE"
    else:
      description = "---"
    return f"{indent} {frame.function}: {description}"

  stack = inspect.stack()
  if limit is not None:
    stack = stack[:limit]

  stack_summary = "\n".join([_format_frame_line(f) for f in stack])
  return stack_summary
