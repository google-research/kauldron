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

"""Utility functions."""

import inspect
import os
import sys
import typing
from typing import Any, Callable
from etils.enp import lazy as enp  # pylint: disable=g-importing-member

TYPE_HINT_CACHING_KEY = "_ktyping_type_hint_cache"


def get_type_name(type_) -> str:
  """Returns a string representation of the type."""

  if type_ is None or type_ is type(None):
    return "None"

  name = (
      getattr(type_, "__name__", None)
      or getattr(type_, "_name", None)
      or getattr(type_, "__forward_arg__", None)
  )
  if name is None:
    origin = typing.get_origin(type_)
    name = getattr(origin, "_name", None)
    if name is None and not inspect.isclass(type_):
      name = type_.__class__.__name__

  args = typing.get_args(type_)
  if args:
    if name == "Literal":
      formatted_args = ", ".join(repr(arg) for arg in args)
    elif name in ("Union", "UnionType"):
      return " | ".join(get_type_name(arg) for arg in args)
    else:
      formatted_args = ", ".join(get_type_name(arg) for arg in args)

    name += f"[{formatted_args}]"

  module = getattr(type_, "__module__", None)
  if module not in (None, "typing", "typing_extensions", "builtins"):
    name = module + "." + name

  return name


def get_type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
  """Return the type hints for the given function with caching."""
  annotations = getattr(fn, TYPE_HINT_CACHING_KEY, None)
  if annotations is None:
    annotations = typing.get_type_hints(fn, include_extras=True)
    setattr(fn, TYPE_HINT_CACHING_KEY, annotations)
  return annotations


def is_running_in_debugger() -> bool:
  """Returns True if running in a debugger."""
  # gettrace is set by the debugger but also by coverage mode in tests.
  gettrace = getattr(sys, "gettrace", lambda: False)
  in_coverage_mode = os.environ.get("COVERAGE_RUN", False)
  return gettrace() and not in_coverage_mode


def get_dtype_str(value) -> str:
  """Get value dtype as a string for any array (np, jnp, tf, torch)."""
  return str(enp.dtype_from_array(value))
