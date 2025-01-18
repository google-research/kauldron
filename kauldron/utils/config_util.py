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

"""Utils for dataclasses."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, ClassVar, TypeVar

from etils import edc
from etils import epy
import jax
from kauldron import konfig

if typing.TYPE_CHECKING:
  from kauldron.train import trainer_lib

_SelfT = TypeVar('_SelfT')


# Use class rather than `object()` to support pickle and better `__repr__`.
class _FakeRefsUnset:
  """Sentinel to mark an object had never applied `update_from_root_cfg`.

  Tracking the original fake reference allow to correctly re-overwrite the
  sub-values when the original config object is updated.

  ```python
  trainer = kd.train.Trainer(
      train_ds=kd.data.Tfds(),
      seed=0,
  )
  assert trainer.train_ds.seed == 0

  dataclasses.replace(trainer, seed=42)
  assert trainer.train_ds.seed == 42  # < Seed is correcly updated.
  ```

  Implementation:

  * When a `UpdateFromRootCfg` is first created, it's `_fake_refs` field is
    set to `_FakeRefsUnset`.
  * When `update_from_root_cfg` is called, the dataclass fields which are
    `_FakeRootCfg` are saved in `_fake_refs`.
  * TODO(epot): When `dataclasses.replace` is called on a `UpdateFromRootCfg`
    object, the `_fake_refs` are propagated to the new object UNLESS the field
    is explicitly overwritten.
  * When `dataclasses.replace` is called on the root `Trainer` and `_fake_refs`
    exists, only the fields defined in `_fake_refs` are overwritten by the
    new resolved `ROOT_CFG_REF` objects.

  Caveat: While `_fake_refs` is not updated when `dataclasses.replace`,
  overwriting field will be a no-op.

  ```python
  trainer = kd.train.Trainer(
      train_ds=kd.data.Tfds(),
      seed=0,
  )

  new_ds = dataclasses.replace(trainer.train_ds, seed=100)
  assert new_ds.seed == 100

  trainer = dataclasses.replace(trainer, train_ds=new_ds)
  assert trainer.train_ds.seed == 0  # < Value overwritten !!!!
  ```
  """


# TODO(epot): Remove `BaseConfig` ?


@dataclasses.dataclass(frozen=True)
class BaseConfig(konfig.WithRef):
  """Base config class.

  For type-checking, fields can be defined like dataclasses:

  ```python
  class Config(BaseConfig):
    workdir: epath.Path
    cleanup: bool = False
    schedule: dict[str, int] = dataclasses.field(default_factory=dict)
  ```

  Fields can be assigned in the `get_config()`:

  * In constructor: `Config(x=1)`
  * As attribute: `config.x = 1`

  Arbitrary fields can be defined (not just the ones defined as dataclass field)
  """

  if not typing.TYPE_CHECKING:
    # pytype fail
    def __init_subclass__(cls, **kwargs):
      super().__init_subclass__(**kwargs)
      cls = dataclasses.dataclass(  # pylint: disable=self-cls-assignment
          frozen=True,
          eq=True,
          kw_only=True,
      )(cls)
      cls = edc.dataclass(cls)  # pylint: disable=self-cls-assignment

  else:
    # For type checking, `__init__` don't require all arguments
    def __init__(self, **kwargs: Any):
      pass

  # TODO(epot): `__repr__` could track objects appearing multiple times in the
  # tree.
  # TODO(epot): pretty_repr should recurse inside `FrozenDict` (custom type)

  def _repr_html_(self) -> str:
    from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return ecolab.highlight_html(repr(self))

  def replace(self: _SelfT, **changes: Any) -> _SelfT:
    return dataclasses.replace(self, **changes)  # pylint: disable=protected-access


@dataclasses.dataclass(frozen=True)
class _FakeRootCfg:
  """Fake root config reference object.

  See `UpdateFromRootCfg` for usage.

  If the field is not set, the value will be copied from the root
  `kd.train.Trainer` object, after it is created.
  """

  parent: _FakeRootCfg | None = None
  name: str = 'cfg'

  def __getattr__(self, name: str) -> Any:
    if name.startswith('_'):
      # Internal / magic methods should raise error as it creates various
      # issues:
      # Raise `AttributeError` when `abc.ABC` check for
      # `hasattr(cls, '__isabstractmethod__')`
      # Context: https://github.com/python/cpython/issues/107580
      # `inspect.unwrap()` check for `__wrapped__`
      return super().__getattribute__('__isabstractmethod__')
    return _FakeRootCfg(parent=self, name=name)

  @classmethod
  def make_fake_cfg(cls) -> trainer_lib.Trainer:
    return cls()  # pytype: disable=bad-return-type

  @property
  def names(self) -> tuple[str, ...]:
    names = []
    curr = self
    while curr is not None:
      names.append(curr.name)
      curr = curr.parent
    return tuple(reversed(names))

  def __repr__(self) -> str:
    qualname = '.'.join(self.names)
    return f'{type(self).__name__}({qualname!r})'

  def __set_name__(self, owner, name):
    if not issubclass(owner, UpdateFromRootCfg):
      raise TypeError(
          '`ROOT_CFG_REF` can only be assigned on subclasses of'
          f' `UpdateFromRootCfg`.\nFor: {owner.__name__}.{name} = {self}'
      )


ROOT_CFG_REF: trainer_lib.Trainer = _FakeRootCfg.make_fake_cfg()


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class UpdateFromRootCfg:
  """Allow child object to be updated with values from the base config.

  For example:

  * `Checkpointer` reuse the `workdir` from the base config.
  * `Evaluator`, `RgnStreams` reuse the `seed` from the base config.

  To use, either:

  * Set your dataclass fields to `ROOT_CFG_REF.xxx` to specify the fields should
    be copied from the base config.
  * Overwrite the `update_from_root_cfg` method, for a custom initialization.

  When using, make sure to also update the `kd.train.Trainer.__post_init__` to
  call
  `update_from_root_cfg`. Currently this not done automatically.

  Example:

  ```python
  @dataclasses.dataclass
  class MyObject:
    workdir: epath.Path = ROOT_CFG_REF.workdir


  root_cfg = kd.train.Trainer(workdir='/path/to/dir')

  obj = MyObject()  # Workdir not set yet

  # Copy the `workdir` from `root_cfg`
  obj = obj.update_from_root_cfg(root_cfg)
  assert obj.workdir == root_cfg.work_dir
  ```

  Attributes:
    _fake_refs: Keep track of the original fake references, so
      `dataclasses.replace` correctly propagate the reference in case the value
      is replaced later on in the Trainer.
    __root_cfg_fields_to_recurse__: List of fields that should be recursively
      updated from the root config. Order is preserved. Is automatically merged
      with all parent classes.
  """

  if not typing.TYPE_CHECKING:
    # TODO(epot): Should wrap `__init__` to check that calling `.replace` with
    # new values remove the ref from fake_refs. A good heuristic would be to
    # check whether `kwargs[name] is getattr(self, name)` to check whether
    # `dataclasses.replace` tried to overwrite the value. But how to get the
    # previous `getattr` ?
    _fake_refs: type[_FakeRefsUnset] | dict[str, _FakeRootCfg] = (
        dataclasses.field(
            default=_FakeRefsUnset,
            compare=False,
            hash=False,
            repr=False,
        )
    )

  __root_cfg_fields_to_recurse__: ClassVar[tuple[str, ...]] = ()

  def update_from_root_cfg(
      self: _SelfT, root_cfg: trainer_lib.Trainer
  ) -> _SelfT:
    """Returns a copy of `self`, potentially with updated values."""
    # Check all fields which are `field: Any = ROOT_CFG_REF.xxx`
    fields_to_replace, fake_refs = self._base_fields(root_cfg)
    # Apply `.update_from_root_cfg()` recursively on fields defined in
    # `__root_cfg_fields_to_recurse__`
    fields_to_replace = self._recurse_fields(root_cfg, fields_to_replace)

    if not fields_to_replace:
      return self
    else:
      return dataclasses.replace(
          self, **fields_to_replace, _fake_refs=fake_refs  # pytype: disable=wrong-keyword-args
      )

  def _base_fields(
      self, root_cfg: trainer_lib.Trainer
  ) -> tuple[dict[str, Any], dict[str, _FakeRootCfg]]:
    """Return the fields to replace."""
    curr_fake_refs = self._fake_refs  # pytype: disable=attribute-error
    fields_to_replace = {}
    fake_refs = {}
    for f in dataclasses.fields(self):
      default = f.default
      if not isinstance(default, _FakeRootCfg):
        continue
      value = getattr(self, f.name)

      # Check all cases:
      # First time `update_from_root_cfg`, replace all `_FakeRootCfg`
      if curr_fake_refs is _FakeRefsUnset:
        if isinstance(value, _FakeRootCfg):
          fake_ref = value  # Replace
        else:
          continue  # Ignore (ROOT_CFG_REF explicitly overwritten)
      else:  # self._fake_refs is a `dict`
        if isinstance(value, _FakeRootCfg):
          # Unexpected, should raise an error ? Indicates the user explicitly
          # set the value to `ROOT_CFG_REF.xxx`
          fake_ref = value  # Always replace ROOT_CFG_REF
        elif f.name in curr_fake_refs:
          # Overwrite the resolved value by the new resolved ROOT_CFG_REF, as
          # the previous value was automatically set by a previous
          # `update_from_root_cfg` call.
          fake_ref = curr_fake_refs[f.name]  # pytype: disable=not-indexable
        else:
          continue  # Ignore (ROOT_CFG_REF explicitly overwritten)

      # value is a fake cfg, should be update
      try:
        new_value = root_cfg
        for attr in fake_ref.names[1:]:  # pytype: disable=attribute-error
          new_value = getattr(new_value, attr)
      except Exception as e:  # pylint: disable=broad-exception-caught
        epy.reraise(
            e,
            prefix=(
                f'Cannot resolve reference {type(self).__name__}.{f.name} ='
                f' {fake_ref}: '
            ),
        )

      # Should auto-recurse into the nested value ?
      # Careful about infinite recursion when the value is top-level
      # `ROOT_CFG_REF`

      fields_to_replace[f.name] = new_value
      fake_refs[f.name] = fake_ref
    return fields_to_replace, fake_refs  # pytype: disable=bad-return-type

  def _recurse_fields(
      self,
      root_cfg: trainer_lib.Trainer,
      fields_to_replace: dict[str, Any],
  ) -> dict[str, Any]:
    """Return the fields to replace."""

    # Detect the fields to update
    all_fields = []
    for cls in type(self).mro():
      if '__root_cfg_fields_to_recurse__' in cls.__dict__:
        all_fields.extend(cls.__root_cfg_fields_to_recurse__)  # pylint: disable=attribute-error

    all_fields_set = set(all_fields)
    if len(all_fields_set) != len(all_fields):
      raise ValueError(
          'Duplicate value in'
          f' `{type(self).__name__}.__root_cfg_fields_to_recurse__` ('
          f'fields from parent classes are automatically merged): {all_fields}'
      )

    # Auto-recurse into fields which are `UpdateFromRootCfg`
    # It shouldn't be possible to have a ref on a parent object as it would
    # create a cycle/infinite recursion. Except the `root_cfg` object which
    # is updated in-place.
    for f in dataclasses.fields(self):
      if not f.init:
        continue
      if f.name in all_fields_set:
        continue
      v = getattr(self, f.name)
      if v is root_cfg:  # Do not recurse into the root config.
        continue
      if isinstance(v, UpdateFromRootCfg):
        all_fields.append(f.name)

    # Replace the fields
    for field_name in all_fields:
      if field_name in fields_to_replace:  # Field already updated.
        new_value = fields_to_replace[field_name]
      else:
        new_value = getattr(self, field_name)

      # Apply `.update_from_root_cfg()` on the field
      fields_to_replace[field_name] = jax.tree.map(
          lambda x: x.update_from_root_cfg(root_cfg),
          new_value,
          is_leaf=lambda x: isinstance(x, UpdateFromRootCfg),
      )

    return fields_to_replace

  def _assert_root_cfg_resolved(self) -> None:
    """Raise an error if one attribute is still a `ROOT_CFG_REF`."""
    return self._assert_root_cfg_resolved_value

  @functools.cached_property
  def _assert_root_cfg_resolved_value(self) -> None:
    for f in dataclasses.fields(self):
      if not isinstance(f.default, _FakeRootCfg):
        continue
      value = getattr(self, f.name)
      if isinstance(value, _FakeRootCfg):
        raise ValueError(
            f'{type(self).__qualname__}.{f.name} is an unresolved'
            f' `ROOT_CFG_REF` value ({value}).\nTo resolve the value, either'
            f' explicitly set `{f.name}` in `__init__`.'
        )
