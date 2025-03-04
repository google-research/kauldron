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

"""Training property."""

from collections.abc import Iterator
import contextlib
import dataclasses
import functools
from typing import TypeVar

from etils import edc
from etils.epy import _internal
from flax import linen as nn

_FnT = TypeVar('_FnT')


@edc.dataclass
@dataclasses.dataclass
class _Context:
  """`is_training` state."""

  is_training_stack: edc.ContextVar[list[bool]] = dataclasses.field(
      default_factory=list
  )

  @property
  def is_training(self) -> bool:
    if not self.is_training_stack:
      raise ValueError(
          'Calling `self.is_training` property, yet `is_training_property=` '
          'kwargs was not set in `.init` / `.apply`.\n'
      )
    return self.is_training_stack[-1]


_context = _Context()


@contextlib.contextmanager
def set_train_property(is_training: bool) -> Iterator[None]:  # pylint: disable=redefined-outer-name
  """Set the `self.is_training` state to the given value."""
  if not isinstance(is_training, bool):
    raise ValueError(
        f'`is_training_property` must be a `bool`. Got: {type(is_training)}:'
        f' {is_training}'
    )
  try:
    _context.is_training_stack.append(is_training)
    yield
  finally:
    _context.is_training_stack.pop()


def _set_train_property(is_training: bool):
  if _context.is_training_stack:
    raise ValueError('Nesting `.apply` / `.init` not supported.')
  return set_train_property(is_training)


def train_property() -> bool:
  """`is_training` property.

  Usage:

  ```python
  class MyModule(nn.Module):
    is_training = kd.nn.train_property()  # No typing annotation here !

    @nn.compact
    def __call__(self, x):
      if self.is_training:
        x = nn.Dropout(.5)(x)

      return x
  ```

  The `is_training` property value has to be set at the top level `.init()` or
  `.apply()` call and will be propagated to all children.

  Alternatively, the value can be set and changed inside any module with the
  `kd.nn.set_train_property(False)` context manager.

  ```python
  model = MyModule()
  model.init(..., is_training_property=True)
  ```

  Returns:
    The `is_training` property
  """
  return property(_is_training)  # pytype: disable=bad-return-type


def _is_training(self: nn.Module) -> bool:
  """`is_training` property."""
  if self.scope is None:
    raise ValueError(
        '`is_training` property can only be called from within `.init` /'
        ' `.apply`.'
    )
  return _context.is_training


@functools.cache
def _mock_flax_to_add_is_training_kwargs() -> None:
  """Add the `is_training=` kwargs to the `.init` / `.apply`."""
  nn.Module.init = _add_is_training_kwargs(nn.Module.init)
  nn.Module.init_with_output = _add_is_training_kwargs(
      nn.Module.init_with_output
  )
  nn.Module.apply = _add_is_training_kwargs(nn.Module.apply)


def _add_is_training_kwargs(fn: _FnT) -> _FnT:
  """Add the `is_training=` kwargs to `fn`."""
  fn = _internal.unwrap_on_reload(fn)  # pylint: disable=protected-access

  @_internal.wraps_with_reload(fn)
  def decorated(*args, is_training_property: bool | None = None, **kwargs):  # pylint: disable=redefined-outer-name
    # Hide the function from the traceback. Supported by Pytest and IPython
    __tracebackhide__ = True  # pylint: disable=unused-variable,invalid-name

    if is_training_property is not None:
      cm = _set_train_property(is_training_property)
    else:
      cm = contextlib.nullcontext()
    with cm:
      return fn(*args, **kwargs)

  return decorated


_mock_flax_to_add_is_training_kwargs()
