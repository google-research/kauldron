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

"""Utility functions."""
from __future__ import annotations

import collections.abc
import dataclasses
import inspect
import sys
import types
import typing
from typing import Any, Callable, Generator

from etils import enp  # pylint: disable=g-importing-member
from kauldron.ktyping import frame_utils
import numpy as np

TYPE_HINT_CACHING_KEY = "_ktyping_type_hint_cache"


# MARK: get_type_name
def get_type_name(type_: Any, full_path: bool = False) -> str:
  """Returns a string representation of the given obj or type."""

  if type_ is None or type_ is type(None):  # pylint: disable=unidiomatic-typecheck
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

  if "." in name:
    # Sometimes the __name__ of a type is a fully qualified name.
    # This happens e.g. for some versions of jax.Array where unfortunately
    # __name__ = 'jaxlib._jax.Array' instead of 'Array'.
    # Here we remove the prefix for readability and only add the path if the
    # user explicitly requested it with `full_path=True`.
    name = name.split(".")[-1]

  args = typing.get_args(type_)
  if args:
    if name == "Literal":
      formatted_args = ", ".join(repr(arg) for arg in args)
    elif name in ("Union", "UnionType"):
      return " | ".join(get_type_name(arg) for arg in args)
    else:
      formatted_args = ", ".join(get_type_name(arg) for arg in args)

    name += f"[{formatted_args}]"

  if full_path:
    if not isinstance(type_, type):
      type_ = type(type_)
    module = getattr(type_, "__module__", None)
    if module not in (None, "typing", "typing_extensions", "builtins"):
      name = module + "." + name

  return name


# MARK: format_value
def format_value(val: Any, truncate: int | None = 40):
  """Returns a string representation of any value with array spec formatting."""
  if enp.ArraySpec.is_array(val):
    # Return type shorthand + ArraySpec for arrays (e.g. np.f32[32 32 3]).
    spec = enp.ArraySpec.from_array(val)
    array_type = _get_array_type_shorthand(val)
    return f"{array_type}.{spec}"

  # Otherwise show a (possibly truncated) repr.
  r = repr(val)
  if len(r) > truncate:
    # Keep the first 36 and the last 1 character of the repr.
    # (This keeps the closing quotes, parenthesis, etc.)
    r = r[: truncate - 4] + "..." + r[-1:]
  return r


def _get_array_type_shorthand(array: Any) -> str:
  if isinstance(array, (np.ndarray, np.generic)):
    return "np"
  elif enp.lazy.has_jax and isinstance(array, enp.lazy.jax.Array):
    return "jax"
  elif enp.lazy.has_tf and isinstance(array, enp.lazy.tf.Tensor):
    return "tf"
  return get_type_name(array)


# MARK: get_type_hints
def get_type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
  """Return the type hints for the given function with caching."""
  annotations = getattr(fn, TYPE_HINT_CACHING_KEY, None)
  if annotations is None:
    annotations = typing.get_type_hints(fn, include_extras=True)
    setattr(fn, TYPE_HINT_CACHING_KEY, annotations)
  return annotations


# MARK: get_dtype_str
def get_dtype_str(value) -> str:
  """Get value dtype as a string for an array or dtype-like object."""
  if enp.lazy.is_array(value):
    return str(enp.lazy.dtype_from_array(value))
  elif inspect.isclass(value) and issubclass(value, np.generic):
    # This is e.g. np.float32, np.int64, np.bool_
    # In that case we return the __name__
    return f"np.{value.__name__}"
  elif isinstance(value, (int, float, complex, bool)):
    return value.__class__.__name__
  else:
    raise TypeError(f"Cannot get dtype string for {value} ({type(value)})")


# MARK: CodeLocation
@dataclasses.dataclass(kw_only=True, frozen=True)
class CodeLocation:
  """Represents the source code location of a typechecked object."""

  description: str
  file: str
  line: int | None
  module_name: str

  @classmethod
  def unknown(cls) -> CodeLocation:
    return cls(
        description="unknown",
        file="<unknown>",
        line=None,
        module_name="<unknown>",
    )

  @classmethod
  def from_any(cls, obj: Any, description: str | None = None) -> CodeLocation:
    """Create a Source from a function or class."""

    if description is None:
      if isinstance(obj, types.FunctionType):
        # Note: We do not support other function types such as
        # types.BuiltinFunctionType, types.WrapperDescriptorType, etc.
        # because the @typechecked decorator does not really apply to them.
        description = f"function '{obj.__name__}'"
      elif isinstance(obj, classmethod):
        description = f"classmethod '{obj.__name__}'"
      elif isinstance(obj, staticmethod):
        description = f"staticmethod '{obj.__name__}'"
      elif isinstance(obj, property):
        description = f"property '{obj.fget.__name__}'"
      elif typing.is_typeddict(obj):
        description = f"TypedDict '{obj.__name__}'"
      elif dataclasses.is_dataclass(obj):
        description = f"dataclass '{obj.__name__}'"
      elif isinstance(obj, type):
        description = f"class '{obj.__name__}'"
      else:
        raise TypeError(f"CodeLocation doesn't support {type(obj)}")

    # For some reason inspect.getsourcelines() unwraps a function before
    # getting the source, but inspect.getfile() does not.
    # We are interested in the file name of the underlying function/class,
    # so we unwrap manually here.
    # This also handles the case of a classmethod or staticmethod for which
    # the inspect.getfile() call would otherwise fail.
    obj = inspect.unwrap(obj)
    module = inspect.getmodule(obj)
    module_name = module.__name__ if module else "<unknown>"
    try:
      filename = inspect.getfile(obj)
      try:
        # Try to get sourcelines if available
        _, lineno = inspect.getsourcelines(obj)
        return cls(
            description=description,
            file=filename,
            line=lineno,
            module_name=module_name,
        )
      except (OSError, TypeError, IOError):
        return cls(
            description=description,
            file=filename,
            line=None,
            module_name=module_name,
        )
    except TypeError:
      if getattr(obj, "__module__", None) == "builtins":
        # Special designation for built-ins
        return cls(
            description=description,
            file="<builtins>",
            line=None,
            module_name=module_name,
        )

      return cls(
          description=description,
          file="<unknown>",
          line=None,
          module_name=module_name,
      )

  @classmethod
  def from_caller(
      cls, stacklevel: int = 0, description="function call"
  ) -> CodeLocation:
    __ktyping_ignore_frame__ = True  # pylint: disable=unused-variable

    caller_frame = frame_utils.get_caller_frame(stacklevel=stacklevel)
    filename = caller_frame.filename
    module_name = inspect.getmodulename(filename) or "<unknown>"
    return cls(
        description=description,
        file=filename,
        line=caller_frame.lineno,
        module_name=module_name,
    )

  def to_str(self) -> str:
    return f"{self.description} at {self.file}:{self.line}"


def get_generator_return_types(gen_fn) -> tuple[Any, Any, Any]:
  """Returns the yield, send and return types of a given generator function."""
  if not inspect.isgeneratorfunction(gen_fn):
    raise TypeError(f"{gen_fn!r} is not a generator function.")

  annotations = get_type_hints(gen_fn)
  gen_type = annotations.get("return", Any)
  gen_type_origin = typing.get_origin(gen_type)  # origin of Any is None
  if gen_type_origin not in (None, Generator, collections.abc.Generator):
    raise TypeError(
        "ktyping only supports typechecking generator functions annotated with"
        f" Generator[...] or Any, not {typing.get_origin(gen_type)}."
    )

  gen_args = typing.get_args(gen_type)
  yield_type, send_type, return_type = Any, None, None
  if not gen_args:
    pass
  elif len(gen_args) == 1:
    (yield_type,) = gen_args
  elif len(gen_args) == 2:
    yield_type, send_type = gen_args
  elif len(gen_args) == 3:
    yield_type, send_type, return_type = gen_args
  else:
    raise TypeError(
        "Invalid number of arguments for Generator: "
        f"Should be less or equal to 3, but was {len(gen_args)}"
    )
  return yield_type, send_type, return_type


def contains_jaxtyping_type(annot: Any) -> bool:
  """Returns True if the annotation contains any jaxtyping types."""
  if "jaxtyping" not in sys.modules:
    # if jaxtyping is not imported all is good
    # (kauldron.typing also depends on jaxtyping)
    return False
  import jaxtyping  # pylint: disable=g-import-not-at-top,unused-import  # pytype: disable=import-error

  if inspect.isclass(annot) and (
      issubclass(annot, jaxtyping.AbstractArray)
      or issubclass(annot, jaxtyping.PyTree)
  ):
    return True

  args = typing.get_args(annot)
  if args:  # Union, tuple, list, ...
    return any(contains_jaxtyping_type(a) for a in args)

  if dataclasses.is_dataclass(annot) or typing.is_typeddict(annot):
    return any(
        contains_jaxtyping_type(a) for a in get_type_hints(annot).values()
    )

  return False
