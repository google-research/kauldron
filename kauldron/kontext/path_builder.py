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

"""Utils to help creating kauldron `Path`."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, Generic, Optional, TypeVar

from etils import epy
import typing_extensions

_T = TypeVar('_T')

_PartItem = str | int | slice


def path_builder_from(prefix: str, cls: type[_T]) -> _T:
  """Create a path builder from a class.

  Used to dynamically create the Keys inside the configs from typed-objects (
  e.g. dataclasses). This adds type-checking and auto-complete (rather than
  using raw `str`).

  Usage:

  ```python
  def get_config():
    batch = kontext.path_builder_from('batch', my_project.Batch)
    out = kontext.path_builder_from('preds', my_project.ModelOutput)

    cfg.model = my_project.MyModel(
        input=batch.image
    )
    cfg.train_losses = {
        'loss': kd.losses.L2(pred=out.logits, target=batch.label)
    }
  ```

  Args:
    prefix: Prefix name of the key (e.g. `batch`, `preds`)
    cls: Class to validate.

  Returns:
    The path builder
  """
  del cls  # Only used for type annotations
  # Maybe could use a `AnnotatedPathBuilder` when known ?
  return DynamicPathBuilder(_PathBuilderState(part=_Root(prefix)))


@dataclasses.dataclass(kw_only=True)
class _PathBuilderState:
  """State of `PathBuilder`.

  Store state in separate object:

  * To avoid issues with `__getattr__`
  * To avoid attribute name collision
  """

  part: _PartRepr
  parent: Optional[_PathBuilderState] = None

  @functools.cached_property
  def parts(self) -> list[_PartRepr]:
    if self.parent is None:
      return [self.part]
    else:
      return self.parent.parts + [self.part]

  @functools.cached_property
  def parts_repr(self) -> str:
    return ''.join(str(p) for p in self.parts)

  def make_child(self, part: _PartRepr, **kwargs: Any) -> _PathBuilderState:
    return self.replace(parent=self, part=part, **kwargs)

  replace = dataclasses.replace


class _PathBuilder:
  """Base path builder class."""

  _state: _PathBuilderState

  def __as_konfig__(self) -> str:
    return str(self)

  def __str__(self) -> str:
    return f'{self._state.parts_repr}'

  def __repr__(self) -> str:
    return f'{type(self).__qualname__}({self})'


class DynamicPathBuilder(_PathBuilder):
  """Util to create paths.

  Can generate arbitrary paths:

  ```python
  p = DynamicPathBuilder()
  str(p.xx['aa'][123].yy) == "DynamicPathBuilder.xx['aa'][123].yy"
  ```
  """

  def __init__(self, state: Optional[_PathBuilderState] = None):
    self._state = state or _PathBuilderState(part=_Root(type(self).__name__))  # pytype: disable=name-error

  def __getattr__(self, name: str) -> DynamicPathBuilder:
    return type(self)(self._state.make_child(_Attribute(name)))

  def __getitem__(self, key: _PartItem) -> DynamicPathBuilder:
    return type(self)(self._state.make_child(_Item(key)))


@dataclasses.dataclass
class _AnnotatedPathBuilderState(_PathBuilderState):
  """State of `AnnotatedPathBuilder`."""

  cls: Optional[type[_AnnotatedPathBuilderState]] = None

  @functools.cached_property
  def hints(self) -> dict[str, Any]:
    if self.cls is None:
      return {}
    else:
      return typing_extensions.get_type_hints(self.cls)


class AnnotatedPathBuilder(_PathBuilder):
  """Util to create paths.

  The annotated version use typing annotations to dynamically infer the
  valid fields.

  ```python
  class A(AnnotatedPathBuilder):
    x: int
    a: A

  a = A()
  str(a.a.a.x) == "A.a.a.x"
  ```
  """

  def __init__(self, state: Optional[_AnnotatedPathBuilderState] = None):
    if state is None:
      state = _AnnotatedPathBuilderState(
          cls=type(self), part=_Root(type(self).__name__)  # pytype: disable=name-error
      )

    self._state = state

  if not typing.TYPE_CHECKING:

    def __getattr__(self, name: str) -> AnnotatedPathBuilder:
      if name not in self._state.hints:
        raise AttributeError(f'{self}.{name}')
      cls = self._state.hints[name]
      if epy.issubclass(cls, _PathBuilder):
        return type(self)(self._state.make_child(_Attribute(name), cls=cls))
      else:
        return type(self)(self._state.make_child(_Attribute(name), cls=None))

  def __dir__(self) -> list[str]:
    """Available attributes (for Colab auto-complete)."""
    return list(self._state.hints)


@dataclasses.dataclass
class _PartRepr(Generic[_T]):
  """Part representation (segment of a path)."""

  value: _T

  def __str__(self) -> str:
    raise NotImplementedError


class _Root(_PartRepr[str]):
  """Root node (`root.xxx`)."""

  def __str__(self) -> str:
    return self.value


class _Attribute(_PartRepr[str]):
  """Attribute node (`xxx.attribute`)."""

  def __str__(self) -> str:
    return f'.{self.value}'


class _Item(_PartRepr[_PartItem]):
  """Item node (`xxx[item]`)."""

  def __str__(self) -> str:
    return f'[{self.value!r}]'
