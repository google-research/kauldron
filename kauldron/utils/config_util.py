# Copyright 2023 The kauldron Authors.
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
from typing import Any, TypeVar

from etils import edc
from kauldron import konfig

if typing.TYPE_CHECKING:
  from kauldron.train import config_lib

_SelfT = TypeVar('_SelfT')

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
  def make_fake_cfg(cls) -> config_lib.Trainer:
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


ROOT_CFG_REF: config_lib.Trainer = _FakeRootCfg.make_fake_cfg()


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
    _REUSE_FROM_ROOT_CFG: Mapping <root_cfg attribute> to <self attribute>
  """

  def update_from_root_cfg(
      self: _SelfT, root_cfg: config_lib.Trainer
  ) -> _SelfT:
    """Returns a copy of `self`, potentially with updated values."""
    fields_to_replace = {}
    for f in dataclasses.fields(self):
      default = f.default
      if not isinstance(default, _FakeRootCfg):
        continue
      value = getattr(self, f.name)
      if not isinstance(value, _FakeRootCfg):
        continue
      # value is a fake cfg, should be update
      new_value = root_cfg
      for attr in value.names[1:]:
        new_value = getattr(root_cfg, attr)
      fields_to_replace[f.name] = new_value
    if not fields_to_replace:
      return self
    else:
      return dataclasses.replace(self, **fields_to_replace)

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
            f' explicitly set `{f.name}` in `__init__`, or call'
            ' `obj.update_from_root_cfg(root_cfg)` to copy the value from the'
            ' root `kd.train.Trainer` object.'
        )
