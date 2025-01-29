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

"""Utils to parially initialize objects."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import dataclasses
import functools
import inspect
import typing
from typing import Any, TypeVar

from etils import edc
from etils import epy
from etils.epy import _internal
from kauldron.utils import immutabledict

# Use lazy-import as this file is imported in Kauldron
with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  import attr
  from kauldron.xm._src import job_params
  from kauldron.xm._src import job_lib
  # pylint: enable=g-import-not-at-top  # pytype: disable=import-error


_T = TypeVar("_T")
_ClsT = TypeVar("_ClsT")


# ------- Init params tracking logic -------


def add_merge_support(cls: _ClsT) -> _ClsT:
  """Allow the class to be merged with other instance.

  When merging two instance, the fields explicitly provided in `__init__` will
  be merged. Field which are `dict` or other merged-supported classes are
  recursed into. If a non-recursed field is defined in both instances, an error
  is raised.

  Examples:

  ```python
  @merge_utils.add_merge_support
  @dataclasses.dataclass
  class A:
    w: int = 0
    x: int = 1
    y: int = 2
    z: dict[str, int] = dataclasses.field(default_factory=dict)


  a0 = A(x=10, z={'a': 30})
  a1 = A(y=20, z={'b': 30})
  assert merge_utils.merge(a0, a1) == A(
      w=0,
      x=10,
      y=20,
      z={
          'a': 30,
          'b': 30,
      },
  )

  merge_utils.merge(A(x=2), A(x=3))  # ValueError: Conflicting `x=` value
  ```

  Args:
    cls: The class to add merge support.

  Returns:
    The decorated class
  """
  # TODO(epot): Allow displaying the origin when there's a value missmatch
  # Extract where the `kxm.Job()` line is defined. How ?
  # Also how to deal with `dataclasses.replace` ?

  cls = _wrap_new(cls)
  cls = _wrap_repr(cls)

  # pylint: disable=protected-access
  cls._kxm_init_sig = functools.cached_property(_kxm_init_sig)
  cls._kxm_init_sig.__set_name__(cls, "_kxm_init_sig")

  cls._kxm_init_kwargs = functools.cached_property(_kxm_init_kwargs)
  cls._kxm_init_kwargs.__set_name__(cls, "_kxm_init_kwargs")
  # pylint: enable=protected-access
  return cls


def _wrap_new(cls: _ClsT) -> _ClsT:
  """`__new__` decorator that save values explicitly given as arg."""

  old_new_fn = _internal.unwrap_on_reload(cls.__new__)

  @_internal.wraps_with_reload(old_new_fn)
  def new_new_fn(cls, *args, **kwargs):
    if old_new_fn is object.__new__:
      self = old_new_fn(cls)
    else:
      self = old_new_fn(cls, *args, **kwargs)

    bound = self._kxm_init_sig.bind_partial(*args, **kwargs)  # pylint: disable=protected-access

    # object.__setattr__ to support `frozen=True`
    object.__setattr__(self, "_kxm_bound", bound)
    return self

  cls.__new__ = new_new_fn
  return cls


def _kxm_init_sig(self) -> inspect.Signature:
  return inspect.signature(self.__init__)


def _kxm_init_kwargs(self: _MergableObj) -> dict[str, Any]:
  return _extract_passed_kwargs(self._kxm_init_sig, self._kxm_bound)  # pylint: disable=protected-access


class _MergableObj:
  """Implementation of objects decorated by `allow_partial_initialization`."""

  _kxm_init_sig: inspect.Signature
  _kxm_bound: inspect.BoundArguments
  _kxm_init_kwargs: dict[str, Any]


# ------- Repr utils -------


@edc.dataclass
@dataclasses.dataclass(kw_only=True)
class _ReprContext:
  _repr_only_init: edc.ContextVar[bool] = False

  @contextlib.contextmanager
  def repr_only_init(self) -> Iterator[None]:
    try:
      self._repr_only_init = True
      yield
    finally:
      self._repr_only_init = False


_repr_context = _ReprContext()
repr_only_init = _repr_context.repr_only_init


def _wrap_repr(cls: _ClsT) -> _ClsT:
  """Decorator around `__repr__` for nicer display inside `kxm.Experiment`."""
  old_repr = _internal.unwrap_on_reload(cls.__repr__)

  @_internal.wraps_with_reload(old_repr)
  def new_repr(self: _MergableObj) -> str:
    # Only activate custom `__repr__` when displayed inside `kxm.Experiment`,...
    if not _repr_context._repr_only_init:  # pylint: disable=protected-access
      if old_repr == epy.pretty_repr:
        # Note this might not support circles but is fine enough
        return epy.pretty_repr_top_level(self, force=True)
      elif attr.has(type(self)):
        # In theory, should only pretty-display when calling inside
        # `epy.pretty_print`, but too complicated to implement
        return epy.pretty_repr_top_level(self, force=True)
      else:
        # Default repr
        return old_repr(self)  # pylint: disable=too-many-function-args
    else:
      return epy.Lines.make_block(
          header=type(self).__qualname__,
          content=self._kxm_init_kwargs,  # pylint: disable=protected-access
      )

  # Overwrite the `__doc__` as it is used in `epy/text_utils.py` to detect
  # if the `__repr__` method is the default one.
  new_repr.__doc__ = "Generated by `kxm`"
  # new_repr._epy_ignore_pretty

  cls.__repr__ = new_repr
  return cls


# ------- Merge logic -------


@typing.overload
def merge(*objs: job_params.JobParams) -> job_lib.Job:
  ...


@typing.overload
def merge(*objs: _T) -> _T:
  ...


def merge(*objs):
  """Merge multiple instances together, recursivelly.

  Instances should previously have been decorated with `add_merge_support`
  See `add_merge_support` for usage.

  Args:
    *objs: Instances to merge

  Returns:
    The merging of all instances
  """
  final_obj = objs[0]
  for obj in objs[1:]:
    final_obj = _merge(
        final_obj,
        obj,
        path=f"({type(final_obj).__name__} / {type(obj).__name__})",
    )
  return final_obj  # pytype: disable=bad-return-type


def _merge(obj0: Any, obj1: Any, *, path: str) -> Any:
  """Merge 2 objects together."""
  # Only the same types can be merged, with the exception of `job_params` that
  # support subclasses.
  if isinstance(obj0, job_params.JobParams) and isinstance(
      obj1, job_params.JobParams
  ):
    pass
  elif _is_mapping(obj0) and _is_mapping(obj1):
    pass
  elif (cls := type(obj0)) == type(obj1):
    # Prevent to merge subclasses, as `_kxm_init_kwargs` only track the
    # cls.__init__ from the decorated class (not the subclasses)
    if hasattr(cls, "_kxm_init_sig") and "_kxm_init_sig" not in cls.__dict__:
      raise TypeError(
          f"Cannot merge subclasses in {path}: {type(obj0)}. Please report an"
          " issue."
      )
  else:
    raise TypeError(
        f"Cannot merge different types in {path}: {obj0!r} with {obj1!r}"
    )

  if hasattr(type(obj0), "_kxm_init_sig"):
    return _merge_obj(obj0, obj1, path=path)
  elif _is_mapping(obj0):
    return _merge_dict(obj0, obj1, path=path)
  elif obj0 == obj1:
    return obj0
  else:
    # If value is specified in multiple places, raise an error.
    # TODO(epot): CLI arguments should always overwrite everything !!! (or
    # actually not for the eval job)
    raise ValueError(
        f"Cannot merge conflicting values in {path}: {obj0!r} with {obj1!r}\n"
        "This likely indicates a conflict between one of:\n"
        " * `--xp.xxx`: Value set to **all** jobs.\n"
        " * `--cfg.xm_job.xxx`: Value set to the train job only.\n"
        " * `--cfg.evals.<my_eval>.run.xxx`: Value set to the eval job only.\n"
        "Value cannot be defined both globally (`--xp.xxx`) and individually"
        " (`--cfg.<>.xxx`)."
    )


def _merge_obj(obj0: _MergableObj, obj1: _MergableObj, *, path: str) -> Any:
  """Merge 2 objects together."""
  final_kwargs = _merge_dict(
      obj0._kxm_init_kwargs, obj1._kxm_init_kwargs, path=path  # pylint: disable=protected-access
  )
  # Could make this a protocol, rather than hardcoding it
  if isinstance(obj0, job_params.JobParams):
    # Filter only the arguments common to `Job`
    available_fields = {f.name for f in dataclasses.fields(job_lib.Job)}
    final_kwargs = {
        k: v for k, v in final_kwargs.items() if k in available_fields
    }
    return job_lib.Job(**final_kwargs)
  else:
    return type(obj0)(**final_kwargs)


def _merge_dict(d0: dict[str, Any], d1: dict[str, Any], *, path: str):
  final_dict = dict(d0)
  for k, v in d1.items():
    if k not in final_dict:  # Non-overlapping key
      final_dict[k] = v
      continue
    # Recurse merge for common keys
    final_dict[k] = _merge(final_dict[k], v, path=f"{path}.{k}")
  return final_dict


def _is_mapping(obj) -> bool:
  return isinstance(obj, (dict, immutabledict.ImmutableDict))


# ------- Inspect utils -------


def _extract_passed_kwargs(
    sig: inspect.Signature, params: inspect.BoundArguments
) -> dict[str, Any]:
  """Extract the `dict` of all kwargs."""
  # Copy arguments and filter the ones not in the original signature (child)
  final_kwargs = {
      k: v for k, v in params.arguments.items() if k in sig.parameters
  }

  for param in sig.parameters.values():
    if param.kind == inspect.Parameter.VAR_KEYWORD:
      additional_kwargs = final_kwargs.pop(param.name, {})
      final_kwargs.update(additional_kwargs)
  return final_kwargs
