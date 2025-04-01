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

"""Defines the @typechecked decorator."""

import functools
import inspect
import typing
from typing import Any, Callable, overload

from kauldron.ktyping import scope
from kauldron.ktyping import typeguard_checkers as check
from kauldron.ktyping import utils
from kauldron.ktyping.internal_typing import UNDEFINED, Undefined  # pylint: disable=g-importing-member,g-multiple-import

# TODO(klausg): global config object?
# a global switch to disable typechecking
# (e.g. for debugging or colab hacking)
TYPECHECKING_ENABLED = True


# TODO(klausg): support annotating dataclasses
# TODO(klausg): support static_methods, classmethods, properties, etc.
# TODO(klausg): support generators

_FnT = typing.TypeVar("_FnT", bound=Callable[..., Any])


@overload
def typechecked(fn: _FnT) -> _FnT:
  ...


@overload
def typechecked(*, new_scope: bool) -> Callable[[_FnT], _FnT]:
  ...


def typechecked(
    fn: _FnT | Undefined = UNDEFINED,
    *,
    new_scope: bool = True,
) -> _FnT | Callable[[_FnT], _FnT]:
  """Decorator to enable runtime type-checking and shape-checking.

  Args:
    fn: the function to decorate
    new_scope: By default, open a new scope for the decorated function. Pass
      False to add typechecking but keep the parent scope.

  Returns:
    The decorated function with type-checking enabled.
  """
  if fn is UNDEFINED:  # called e.g. as `@typechecked(new_scope=False)`
    return functools.partial(typechecked, new_scope=new_scope)
  assert isinstance(fn, Callable)

  if hasattr(
      fn, "__wrapped__"
  ):  # TODO(klausg): allow non-innermost decorators?
    raise AssertionError("@typechecked should be the innermost decorator")

  scope_class = scope.ShapeScope if new_scope else scope.TransparentScope
  sig = inspect.signature(fn)

  @functools.wraps(fn)
  def _typechecked_wrapper(*args, **kwargs):
    # Hide the function from the traceback. Supported by Pytest and IPython
    __tracebackhide__ = True  # pylint: disable=unused-variable,invalid-name

    if not TYPECHECKING_ENABLED:
      # typchecking disabled globally or locally -> just return fn(...)
      return fn(*args, **kwargs)

    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    annotations = utils.get_type_hints(fn)
    annotated_args = {
        k: (v, annotations[k])
        for k, v in bound_args.arguments.items()
        if k in annotations
    }

    with scope_class(function=fn, bound_args=bound_args.arguments):
      # check argument types against annotations
      # (inlined here to avoid any additional frames in the traceback)
      for argname, (value, annot) in annotated_args.items():
        check.assert_not_never(fn, annot)
        try:
          check.check_type(value, annot)
        except check.TypeCheckError as exc:
          type_name = utils.get_type_name(value)
          exc.append_path_element(f'argument "{argname}" ({type_name})')
          raise

      # call the decorated function
      value = fn(*args, **kwargs)

      # check return type against annotations
      annot = annotations.get("return", Any)
      check.assert_not_noreturn(fn, annot)
      try:
        check.check_type(value, annot)
      except check.TypeCheckError as exc:
        type_name = utils.get_type_name(value)
        exc.append_path_element(f"return value ({type_name})")
        raise

      # Finally return the return value.
      return value

  return _typechecked_wrapper
