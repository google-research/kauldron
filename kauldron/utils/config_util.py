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

import dataclasses
import typing
from typing import Any, TypeVar

from etils import edc
from kauldron import konfig

_SelfT = TypeVar('_SelfT')


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

  Fields can be assigned:

  * In constructor: `Config(x=1)`
  * As attribute: `config.x = 1`

  Arbitrary fields can be defined (not just the ones defined as dataclass field)
  ```
  """

  if not typing.TYPE_CHECKING:
    # pytype fail
    def __init_subclass__(cls, **kwargs):
      super().__init_subclass__(**kwargs)
      cls = dataclasses.dataclass(  # pylint: disable=self-cls-assignment
          frozen=True,
          eq=True,
          kw_only=True,
          init=False,
      )(cls)
      cls = edc.dataclass(cls)  # pylint: disable=self-cls-assignment

  def __init__(self, **kwargs: Any):
    values = dict(kwargs)
    for f in dataclasses.fields(self):
      if f.name in values:  # Field explicitly passed
        continue
      elif f.default is not dataclasses.MISSING:
        values[f.name] = f.default
      elif f.default_factory is not dataclasses.MISSING:
        values[f.name] = f.default_factory()
    for k, v in values.items():
      object.__setattr__(self, k, v)

    if hasattr(self, '__post_init__'):
      self.__post_init__()

  def __repr__(self) -> str:
    return repr(konfig.ConfigDict(self._field_values))

  def _repr_html_(self) -> str:
    return konfig.ConfigDict(self._field_values)._repr_html_()  # pylint: disable=protected-access

  def replace(self: _SelfT, **changes: Any) -> _SelfT:
    return type(self)(**self._field_values | changes)

  if typing.TYPE_CHECKING:

    def __getattr__(self, name: str) -> Any:
      super().__getattribute__(name)

  @property
  def _field_values(self) -> dict[str, Any]:
    new_values = dict(self.__dict__)
    for f in dataclasses.fields(self):  # Descriptors are not in `__dict__`
      if hasattr(self, f.name):
        new_values[f.name] = getattr(self, f.name)
    return new_values
