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

"""Intermediate utils."""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Optional, TypeVar

from etils import edc
from kauldron.klinen.collections import Collection
from typing_extensions import Annotated

if typing.TYPE_CHECKING:
  from kauldron.klinen import module as module_lib


_T = TypeVar('_T')
_SelfT = TypeVar('_SelfT')

_IS_INTERMEDIATE = object()

if typing.TYPE_CHECKING:
  # TODO(b/254514368): Remove hack
  class _IntermediateMeta(type):

    def __getitem__(cls, value):
      return value

  class Intermediate(metaclass=_IntermediateMeta):
    pass

else:
  Intermediate = Annotated[_T, _IS_INTERMEDIATE]  # pytype: disable=invalid-typevar


KEY_PREFIX = '_attribute__'


@dataclasses.dataclass
class IntermediateDescriptor:
  """Descriptor to read-write individual contextvar."""

  field: dataclasses.Field[Any]
  objtype: type[Any] = dataclasses.field(init=False)
  attribute_name: str = dataclasses.field(init=False)

  @classmethod
  def from_field(
      cls, field: dataclasses.Field[Any], hint: edc.helpers.Hint
  ) -> IntermediateDescriptor:
    if field.init:
      raise ValueError(
          '`knn.Intermediate[T]` fields should be'
          f' `dataclasses.field(init=False)` for `{hint}`'
      )
    return cls(field)

  @property
  def _collection_name(self) -> str:
    """Name of the attribute in the `.sow('intermediates', name)` collection."""
    return f'{KEY_PREFIX}{self.attribute_name}'

  @property
  def _default(self) -> Any:
    """Default value."""
    if self.field.default is not dataclasses.MISSING:
      default = self.field.default
    elif self.field.default_factory is not dataclasses.MISSING:
      default = self.field.default_factory()
    else:
      raise AttributeError(
          f'{self.objtype.__name__!r} object cannot access intermediate'
          f' attribute {self.attribute_name!r}. Attribute was not set during'
          ' the call.'
      )
    return default

  def __set_name__(self, objtype: type[module_lib.Module], name: str) -> None:
    """Bind the descriptor to the class (PEP 487)."""
    self.objtype = objtype
    self.attribute_name = name

  def __get__(
      self,
      obj: Optional[module_lib.Module],
      objtype: Optional[type[module_lib.Module]] = None,
  ):
    """`x = module.my_intermediate`."""

    if obj is None:
      return self

    if not obj.scope:
      raise AttributeError(
          f'Intermediate field `{objtype.__name__}.{self.attribute_name}`'
          ' can only be accessed inside module functions. Use '
          '`model.capture_intermediate()` to access the intermediate values.'
      )

    if not obj.scope.has_variable(
        Collection.INTERMEDIATES, self._collection_name
    ):
      obj.sow(
          Collection.INTERMEDIATES,
          self._collection_name,
          self._default,
          reduce_fn=_replace_previous_value,
      )
    return obj.get_variable(Collection.INTERMEDIATES, self._collection_name)

  def __set__(self, obj: module_lib.Module, value: Any) -> None:
    """`module.my_intermediate = x`."""

    if not obj.scope:
      if not hasattr(obj, '_kd_init_finished'):
        # No-op during `__init__`.
        # This is to support `dataclasses.field(default=...)`
        return
      raise AttributeError(
          f'Intermediate field `{type(obj).__name__}.{self.attribute_name}`'
          ' can only be set inside module functions.'
      )
    obj.sow(
        Collection.INTERMEDIATES,
        self._collection_name,
        value,
        reduce_fn=_replace_previous_value,
    )


def _replace_previous_value(old: Any, new: _T) -> _T:
  """Merge function for `.sow` which always overwrite the value."""
  del old
  return new


def setup_cls(cls: type[module_lib.Module]) -> None:
  """Wraps `Intermediate[T]` fields in `IntermediateDescriptor` descriptors."""
  # Replace fields annotated with `Intermediate[T]` by their descriptor
  edc.helpers.wrap_new(
      cls,
      descriptor_infos=[
          edc.helpers.DescriptorInfo(
              annotation=Intermediate,
              descriptor_fn=IntermediateDescriptor.from_field,
          )
      ],
  )
