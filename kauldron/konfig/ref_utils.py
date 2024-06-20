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

"""Ref utils."""

from __future__ import annotations

import copy
import functools
import operator
import typing
from typing import Any, Callable, ParamSpec, TypeVar

from kauldron.konfig import configdict_base
import ml_collections

_T = TypeVar('_T')
_P = ParamSpec('_P')
_FnT = TypeVar('_FnT')
_SelfT = TypeVar('_SelfT')

config_dict = ml_collections.config_dict.config_dict  # pylint: disable=protected-access


class WithRef:
  """Protocol to better access lazy fields.

  This only add the `.ref` property for better type-checking & auto-complete.
  This do not modify the object behavior.
  """

  @property
  def ref(self: _SelfT) -> _SelfT:
    """Lazy reference access.

    Before:

    ```python
    cfg.get_ref('workdir')
    ```

    After:

    ```python
    cfg.ref.workdir
    ```

    Raises:
      RuntimeError: When used outside of a `konfig.ConfigDict` context.
    """
    raise RuntimeError(
        f'{type(self).__name__}.ref should only be called when objects is'
        ' imported from `konfig.imports()`'
    )


class _FieldReference(ml_collections.FieldReference):
  """Similar to ml_collections.FieldReference, but supports more control flow.

  This allows to lazy-evaluate conditions and other comparison statement.

  ```python
  cfg = konfig.ConfigDict(x=1)

  # y lazily evaluated
  cfg.y = (cfg.ref.x * 10) > 10

  # Attribute,...
  cfg.model = Model()
  cfg.model.ref.num_layers  # lazy property access
  cfg.model.ref.some_function()  # Lazy function call
  ```
  """

  # =========== Lazy resolve ===========

  # TODO(epot): Currently, this does not support if the config get
  # copied/duplicated. Instead, should propagate the closures.

  # TODO(epot): `FieldReference` has some risk of colision (e.g. `.get()`
  # resolve)

  def __setattr__(self, name: str, value: Any):
    if name.startswith('_'):
      super().__setattr__(name, value)
    else:
      # Could eventually add support for `__setattr__`, but not clear how
      # this would work when trying to read the attribute afterward.
      raise AttributeError(
          f'FieldReference do not support setting attribute ({name}).'
      )

  def __getattr__(self, name: str) -> _FieldReference:
    return self._apply_op(operator.attrgetter(name), new_type=object)

  def __getitem__(self, name: Any) -> Any:
    def fn_flow(value, name):
      return value[name]

    return self._apply_op(fn_flow, name, new_type=object)

  def __call__(self, *args, **kwargs):
    # Flatten/repack args/kwargs
    num_args = len(args)
    kwargs_keys = list(kwargs.keys())

    def fn_flow(value, *flat_args):
      final_args = flat_args[:num_args]
      final_kwargs = dict(zip(kwargs_keys, flat_args[num_args:]))
      return value(*final_args, **final_kwargs)

    return self._apply_op(fn_flow, *args, *kwargs.values(), new_type=object)

  # Overwritte the current unituitive behavior which evaluate "ref == x" as
  # "ref.get() == x"

  def __eq__(self, other):
    return self._apply_op(operator.eq, other)

  def __ne__(self, other):
    return self._apply_op(operator.ne, other)

  def __gt__(self, other):
    return self._apply_op(operator.gt, other)

  def __lt__(self, other):
    return self._apply_op(operator.lt, other)

  def __ge__(self, other):
    return self._apply_op(operator.ge, other)

  def __le__(self, other):
    return self._apply_op(operator.le, other)

  def __iter__(self):
    # ml_collections.FieldReference implement `__getitem__` so is automatically
    # iterable, but this lead to infinite loop.
    raise TypeError(f'{type(self).__name__} object is not iterable')

  # =========== Other functions ===========

  def __init__(
      self,
      default: Any,
      field_type: Any | None = None,
      op: Any | None = None,
      required: bool = False,
  ):
    self._value = None
    super().__init__(default, field_type=field_type, op=op, required=required)

  def __repr__(self) -> str:
    try:
      value = self.get()
    except Exception:  # pylint: disable=broad-exception-caught
      return 'FieldReference(<Unresolved>)'
    else:
      return f'FieldReference({value!r})'

  # TODO(epot): Should try yo merge the fixes back to `ml_collections`

  def _apply_op(self, fn, *args, new_type: Any = None):
    """Overwrite `_apply_op` to return `FieldReference` from `konfig`."""
    args = [config_dict._safe_cast(arg, self._field_type) for arg in args]  # pylint: disable=protected-access
    if new_type is None:
      new_type = self._field_type
    return _FieldReference(
        self,
        field_type=new_type,
        op=config_dict._Op(fn, args),  # pylint: disable=protected-access
    )

  def get(self):
    """Gets the value of the `FieldReference` object.

    This will dereference `_pointer` and apply all ops to its value.

    Returns:
      The result of applying all ops to the dereferenced pointer.

    Raises:
      RequiredValueError: if `required` is True and the underlying value for the
          reference is False.
    """
    if self._required and self._value is None:
      raise config_dict.RequiredValueError(
          'None value found in required reference'
      )

    value = config_dict._get_computed_value(self._value)  # pylint: disable=protected-access
    for op in self._ops:
      # Dereference any FieldReference objects
      args = [config_dict._get_computed_value(arg) for arg in op.args]  # pylint: disable=protected-access
      value = op.fn(value, *args)
      value = config_dict._get_computed_value(value)  # pylint: disable=protected-access
    return value


@typing.overload
def ref_fn(fn: _FnT) -> _FnT:
  ...


@typing.overload
def ref_fn(fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
  ...


def ref_fn(fn, *args, **kwargs):
  r"""Wrap a function for lazy-evaluation.

  Example:

  ```python
  @konfig.ref_fn
  def _join_parts(parts):
    return '/'.join(parts)

  cfg = konfig.ConfigDict()
  cfg.parts = ['a', 'b', 'c']
  cfg.joined_parts = _join_parts(cfg.ref.parts)

  assert cfg.joined_parts == 'a/b/c'

  cfg.parts = ['d', 'e', 'f']
  assert cfg.joined_parts == 'd/e/f'
  ```

  When `cfg.joined_parts` is accessed, the `_join_parts` lazy function will
  be executed.

  Can also be applied inline:

  ```python
  cfg.joined_parts = konfig.ref_fn('\'.join, cfg.ref.parts)
  ```

  Args:
    fn: The function to lazy-evaluate
    *args: Args and kwargs to pass to the function
    **kwargs: Args and kwargs to pass to the function

  Returns:
    The `FieldReference` to be assigned.
  """
  if args or kwargs:
    return ref_fn(fn)(*args, **kwargs)

  @functools.wraps(fn)
  def decorated(*args, **kwargs):
    # Flatten/repack args/kwargs
    num_args = len(args)
    kwargs_keys = list(kwargs.keys())

    def new_fn(_, *flat_args):
      final_args = flat_args[:num_args]
      final_kwargs = dict(zip(kwargs_keys, flat_args[num_args:]))
      return fn(*final_args, **final_kwargs)

    field = _FieldReference(None, field_type=object)
    field = field._apply_op(new_fn, *args, *kwargs.values())  # pylint: disable=protected-access
    return field

  return decorated


# TODO(epot): Ideally, it should be nativelly supported by
# `cfg2 = cfg.oneway_ref`. For this, it would require a ConfigDictRef which
# support attribut assignement.
def ref_copy(cfg: _T) -> _T:
  """One-way recursive copy of the `ConfigDict`.

  Usage:

  ```python
  train_ds = konfig.ConfigDict(dict(
      src=dict(
          name='mnist',
          split='train',
      ),
      shuffle=True,
      batch_size=32,
  ))

  test_ds = konfig.ref_copy(train_ds)

  # Updating the copy won't update `train_ds`
  test_ds.src.split = 'test'
  test_ds.shuffle = False

  # Updating the original config update the ref_copy
  train_ds.batch_size = 128
  assert test_ds.batch_size == 128
  ```

  Caveat: If the original dict is updated after being copied, only the existing
  non-dict attributes will be propagated to the copy.

  ```python
  # New attribute
  train_ds.new_attribute = 123
  assert not hasattr(test_ds, 'new_attribute')

  # Overwritte dict
  train_ds.src = dict(
      name='imagenet',
      split='train',
  )
  assert test_ds.src.name == 'mnist'
  ```

  Args:
    cfg: The dict to recursivelly copy

  Returns:
    The copied config.
  """
  if not isinstance(cfg, ml_collections.ConfigDict):
    raise ValueError(
        f'ref_copy can only be applied on ConfigDict. Not {type(cfg)}'
    )
  cfg_copy = configdict_base.ConfigDict()
  for key, value in cfg.items(preserve_field_references=True):
    match value:
      case ml_collections.FieldReference():
        value = value.identity()  # One-way reference
      case ml_collections.ConfigDict():
        value = ref_copy(value)  # Recursive copy
      case _:
        value = cfg.get_oneway_ref(key)
    cfg_copy[key] = value
  return cfg_copy


def _deepcopy(self, memo: dict[int, Any]):
  """Deepcopy support for FieldReference."""
  # Should we add the new FieldReference to the memo ?
  new = type(self)(
      default=copy.deepcopy(self._value, memo),  # pylint: disable=protected-access
      field_type=self._field_type,  # pylint: disable=protected-access
      required=self._required,  # pylint: disable=protected-access
  )
  new._ops = self._ops  # pylint: disable=protected-access
  return new


# Mock the base class to support deepcopy
ml_collections.FieldReference.__deepcopy__ = _deepcopy


# Mock the base class to support ref on childs
ml_collections.ConfigDict.ref = property(_FieldReference)
