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
  is_training: edc.ContextVar[bool | None] = None


_context = _Context()


# TODO(epot): Allow nesting is_training (e.g. use a pre-trained encoder inside
# another model)
@contextlib.contextmanager
def _set_training(is_training: bool) -> Iterator[None]:  # pylint: disable=redefined-outer-name
  """Update the `is_training` state."""
  if _context.is_training is not None:
    raise ValueError('Nesting `.apply` / `.init` not supported.')

  try:
    _context.is_training = is_training
    yield
  finally:
    _context.is_training = None


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

  is_training = _context.is_training
  if is_training is None:
    raise ValueError(
        'Calling `self.is_training` property, yet `is_training_property=` '
        'kwargs was not set in `.init` / `.apply`'
    )
  return is_training


@functools.cache
def _mock_flax_to_add_is_training_kwargs() -> None:
  """Add the `is_training=` kwargs to the `.init` / `.apply`."""
  nn.Module.init = _add_is_training_kwargs(nn.Module.init)
  nn.Module.apply = _add_is_training_kwargs(nn.Module.apply)


def _add_is_training_kwargs(fn: _FnT) -> _FnT:
  """Add the `is_training=` kwargs to `fn`."""
  fn = _internal.unwrap_on_reload(fn)  # pylint: disable=protected-access

  @functools.wraps(fn)
  def decorated(*args, is_training_property: bool | None = None, **kwargs):  # pylint: disable=redefined-outer-name
    if is_training_property is not None:
      cm = _set_training(is_training_property)
    else:
      cm = contextlib.nullcontext()
    with cm:
      return fn(*args, **kwargs)

  return decorated


_mock_flax_to_add_is_training_kwargs()
