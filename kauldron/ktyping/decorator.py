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

"""Defines the @typechecked decorator."""

import dataclasses
import functools
import inspect
import typing
from typing import Any, Callable, overload
from etils import epy
from kauldron.ktyping import config
from kauldron.ktyping import errors
from kauldron.ktyping import scope
from kauldron.ktyping import typeguard_checkers as check
from kauldron.ktyping import utils
from kauldron.ktyping.internal_typing import MISSING, Missing  # pylint: disable=g-importing-member, g-multiple-import


# TODO(klausg): support generators

_WrappableT = typing.TypeVar("_WrappableT")
_FnT = typing.TypeVar("_FnT", bound=Callable[..., Any])


# MARK: TypeCheckedPartial
@dataclasses.dataclass(kw_only=True, frozen=True)
class _TypeCheckedPartial(epy.ContextManager):
  """Class to support both decorator and context manager usage."""

  new_scope: bool = True

  def __call__(self, obj: _WrappableT) -> _WrappableT:
    source = utils.CodeLocation.from_any(obj)
    # The decorator case.
    # Dataclasses
    if dataclasses.is_dataclass(obj):
      return _wrap_dataclass_with_typechecks(obj, new_scope=self.new_scope)
    # Classmethods
    elif isinstance(obj, classmethod):
      return classmethod(
          _wrap_fn_with_typechecks(
              obj.__func__,
              new_scope=self.new_scope,
              source=source,
          )
      )
    # Staticmethods
    elif isinstance(obj, staticmethod):
      return staticmethod(
          _wrap_fn_with_typechecks(
              obj.__func__,
              new_scope=self.new_scope,
              source=source,
          )
      )
    elif isinstance(obj, property):
      return _wrap_property_with_typechecks(obj, new_scope=self.new_scope)
    # Generator functions
    # TODO(klausg): support classmethod / staticmethod generator functions
    elif inspect.isgeneratorfunction(obj):
      return _wrap_generator_with_typechecks(
          obj, new_scope=self.new_scope, source=source
      )
    # Functions and regular methods
    elif inspect.isfunction(obj):
      return _wrap_fn_with_typechecks(
          obj, new_scope=self.new_scope, source=source
      )
    else:
      raise ValueError(f"Unsupported object type: {type(obj)}")

  def scope_class(
      self,
  ) -> type[scope.ShapeScope] | type[scope.TransparentScope]:
    return scope.ShapeScope if self.new_scope else scope.TransparentScope

  def __contextmanager__(self):
    """This is called when typechecked() is used as a context manager."""

    # stacklevel 3 to ignore the indirection by epy.ContextManager
    source = utils.CodeLocation.from_caller(
        description="typechecked context", stacklevel=3
    )
    ScopeClass = scope.ShapeScope if self.new_scope else scope.TransparentScope  # pylint: disable=invalid-name

    with ScopeClass(source=source, stacklevel=3):
      yield


# MARK: typechecked
@overload
def typechecked(fn: _WrappableT | Missing = MISSING) -> _WrappableT:
  ...


@overload
def typechecked(*, new_scope: bool = True) -> _TypeCheckedPartial:
  ...


def typechecked(
    fn=MISSING,
    *,
    new_scope=True,
):
  """Decorator to enable runtime type-checking and shape-checking.

  Args:
    fn: the function to decorate
    new_scope: By default, open a new scope for the decorated function. Pass
      False to add typechecking but keep the parent scope.

  Returns:
    The decorated function with type-checking enabled.
  """
  tchecked = _TypeCheckedPartial(new_scope=new_scope)

  if fn is not MISSING:
    # called as a decorator
    assert not isinstance(fn, Missing)  # help pytype understand
    return tchecked(fn)

  # This can mean several things:
  # - called e.g. as `@typechecked(new_scope=False)`
  # - called as `with typechecked():`
  return tchecked


# MARK: wrap fn
def _wrap_fn_with_typechecks(
    fn: _WrappableT, new_scope: bool, source: utils.CodeLocation | None = None
) -> _WrappableT:
  """Wraps the given function with typechecking logic."""
  sig = inspect.signature(fn)

  @functools.wraps(fn)
  def _typechecked_wrapper(*args, **kwargs):
    # Hide the function from the traceback. Supported by Pytest and IPython
    __tracebackhide__ = True  # pylint: disable=unused-variable,invalid-name
    # Mark this function as a wrapper to support frame_utils.has_active_scope()
    __ktyping_wrapper__ = True  # pylint: disable=unused-variable

    if not config.get_config(source).typechecking_enabled:
      # typchecking disabled globally or locally -> just return fn(...)
      return fn(*args, **kwargs)

    bound_args = sig.bind(*args, **kwargs)
    non_default_args = {k for k in bound_args.arguments}
    bound_args.apply_defaults()
    bound_args = bound_args.arguments
    annotations = utils.get_type_hints(fn)
    annotated_args = {
        k: (v, annotations[k])
        for k, v in bound_args.items()
        if k in annotations
    }
    default_args = [k for k in bound_args if k not in non_default_args]

    with scope.create_scope_for(
        obj=fn,
        fstring_locals=bound_args,
        arguments=bound_args,
        annotations=annotations,
        default_args=default_args,
        transparent=not new_scope,
        source=source,
    ) as sscope:
      # check argument types against annotations
      # (inlined here to avoid any additional frames in the traceback)
      for argname, (value, annot) in annotated_args.items():
        check.assert_not_never(fn, annot)
        try:
          check.check_type_internal(value, annot)
        except errors.TypeCheckError as exc:
          value_str = utils.format_value(value)
          errors.KTypeCheckError.raise_from_exc(
              exc=exc,
              scope=sscope,
              additional_path_element=f"argument {argname} = {value_str}",
              maybe_highlight=argname,
          )

      # call the decorated function
      value = fn(*args, **kwargs)

      # check return type against annotations
      annot = annotations.get("return", Any)
      sscope.return_value = value
      check.assert_not_noreturn(fn, annot)

      try:
        check.check_type_internal(value, annot)
      except errors.TypeCheckError as exc:
        value_str = utils.format_value(value)
        errors.KTypeCheckError.raise_from_exc(
            exc=exc,
            scope=sscope,
            additional_path_element=f"return value {value_str}",
        )

      # Finally return the return value.
      return value

  return _typechecked_wrapper


# MARK: wrap generator
def _wrap_generator_with_typechecks(
    gen_fn: _FnT,
    new_scope: bool,
    source: utils.CodeLocation | None = None,
) -> _FnT:
  """Wraps the given generator function with typechecking logic."""

  sig = inspect.signature(gen_fn)

  @functools.wraps(gen_fn)
  def _typechecked_generator_wrapper(*args, **kwargs):
    # Hide the function from the traceback. Supported by Pytest and IPython
    __tracebackhide__ = True  # pylint: disable=unused-variable,invalid-name
    # Mark this function as a wrapper to support frame_utils.has_active_scope()
    __ktyping_wrapper__ = True  # pylint: disable=unused-variable

    if not config.get_config(source).typechecking_enabled:
      result = yield from gen_fn(*args, **kwargs)
      return result

    bound_args = sig.bind(*args, **kwargs)
    non_default_args = {k for k in bound_args.arguments}
    bound_args.apply_defaults()
    bound_args = bound_args.arguments
    annotations = utils.get_type_hints(gen_fn)
    annotated_args = {
        k: (v, annotations[k])
        for k, v in bound_args.items()
        if k in annotations
    }
    default_args = [k for k in bound_args if k not in non_default_args]

    with scope.create_scope_for(
        obj=gen_fn,
        fstring_locals=bound_args,
        arguments=bound_args,
        annotations=annotations,
        default_args=default_args,
        transparent=not new_scope,
        source=source,
    ) as sscope:
      # Check argument types
      for argname, (value, annot) in annotated_args.items():
        check.assert_not_never(gen_fn, annot)

        try:
          check.check_type_internal(value, annot)
        except errors.TypeCheckError as exc:
          value_str = utils.format_value(value)
          errors.KTypeCheckError.raise_from_exc(
              exc=exc,
              scope=sscope,
              additional_path_element=f"argument {argname} = {value_str}",
              maybe_highlight=argname,
          )

      # Run the generator function
      result = yield from gen_fn(*args, **kwargs)
      # TODO(klausg): check yield type
      # TODO(klausg): check send type
      # checking these will be messy because yield from is quite complex
      # Also we only want to check the the first yield value and send value
      # for efficiency reasons.

      # Check return type against annotations
      _, _, return_type = utils.get_generator_return_types(gen_fn)
      check.assert_not_noreturn(gen_fn, return_type)
      try:
        check.check_type_internal(result, return_type)
      except errors.TypeCheckError as exc:
        value_str = utils.format_value(result)
        errors.KTypeCheckError.raise_from_exc(
            exc=exc,
            scope=sscope,
            additional_path_element=f"return value {value_str}",
        )
      return result

  return _typechecked_generator_wrapper


# MARK: wrap property
def _wrap_property_with_typechecks(p: property, new_scope: bool) -> property:
  """Wraps the given property with typechecking logic."""
  fget = fset = fdel = None
  # Getter
  if p.fget is not None:
    pname = getattr(p.fget, "__name__", "<unknown>")
    get_source = utils.CodeLocation.from_any(
        p.fget, description=f"property '{pname}'"
    )
    fget = _wrap_fn_with_typechecks(
        p.fget, new_scope=new_scope, source=get_source
    )
  # Setter
  if p.fset is not None:
    pname = getattr(p.fset, "__name__", "<unknown>")
    set_source = utils.CodeLocation.from_any(
        p.fset, description=f"property '{pname}'"
    )
    fset = _wrap_fn_with_typechecks(
        p.fset, new_scope=new_scope, source=set_source
    )
  # Deleter
  if p.fdel is not None:
    pname = getattr(p.fdel, "__name__", "<unknown>")
    del_source = utils.CodeLocation.from_any(
        p.fdel, description=f"property '{pname}'"
    )
    fdel = _wrap_fn_with_typechecks(
        p.fdel, new_scope=new_scope, source=del_source
    )
  return property(fget, fset, fdel)


# MARK: wrap dataclass
def _wrap_dataclass_with_typechecks(
    dcls: type[Any], new_scope: bool
) -> type[Any]:
  """Wraps the given dataclass with typechecking logic."""
  init = dcls.__init__

  @functools.wraps(init)
  def _wrapped_init(self, *args, **kwargs):
    # Hide the function from the traceback. Supported by Pytest and IPython
    __tracebackhide__ = True  # pylint: disable=unused-variable,invalid-name
    # Mark this function as a wrapper to support frame_utils.has_active_scope()
    __ktyping_wrapper__ = True  # pylint: disable=unused-variable

    init(self, *args, **kwargs)

    source = utils.CodeLocation.from_any(dcls)
    if not config.get_config(source).typechecking_enabled:
      return

    if self.__class__.__init__ is not dcls.__init__:
      # Only trigger typechecking for the "top-level" __init__ call.
      # E.g. do not trigger for super().__init__(...) calls from subclasses.
      return

    # check the dataclass fields against their annotations
    fields = dataclasses.fields(dcls)
    field_values = {
        field.name: getattr(self, field.name, MISSING) for field in fields
    }
    field_annotations = {field.name: field.type for field in fields}
    fields_still_default = [
        field.name
        for field, value in zip(fields, field_values.values())
        if value is field.default
    ]

    with scope.create_scope_for(
        obj=dcls,
        transparent=not new_scope,
        arguments=field_values,
        annotations=field_annotations,
        default_args=fields_still_default,
    ) as sscope:
      for field_name, field_value in field_values.items():
        try:
          check.check_type_internal(field_value, field_annotations[field_name])
        except errors.TypeCheckError as exc:
          errors.KTypeCheckError.raise_from_exc(
              exc=exc,
              scope=sscope,
              additional_path_element=f"value of field {field_name!r}",
              maybe_highlight=field_name,
          )

  dcls.__init__ = _wrapped_init
  dcls.__ktyping_wrapped__ = True

  return dcls
