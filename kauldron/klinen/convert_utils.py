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

"""Convert utils."""

import functools
from typing import TypeVar

from etils import epy
from kauldron.klinen import module

_ClsT = TypeVar('_ClsT')  # , bound=type[nn.Module])


@functools.cache
def convert(cls: _ClsT) -> _ClsT:
  """Decorator that convert a flax class into klinen.

  This is equivalent to inheriting from `klinen.Module`, but the new
  attributes/properties (`.init_bind`,...) are not detected by static type
  checker, so inheritance is prefered.
  This function allow quick usage on Colab.

  Args:
    cls: The flax module class

  Returns:
    The decorated klinen module class
  """

  @epy.wraps_cls(cls)
  class NewModule(cls, module.Module):
    pass

  return NewModule
