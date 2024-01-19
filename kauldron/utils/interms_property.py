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

"""Interms property."""
from __future__ import annotations

from flax import linen as nn


def interms_property():
  """`interms` property that makes storing intermediates more convenient.

  Usage:

  ```python
  class MyModule(nn.Module):
    interms = kd.nn.interms_property()  # No typing annotation here !

    @nn.compact
    def __call__(self, x):
      h = nn.Dense(12)(x)

      # append to intermediates.
      self.interms["hidden"] = h

      # Access with `interms.path.to.module.hidden[0]`  (no `__call__`)
      # Setting interms like above is equivalent to using:
      # self.sow("intermediates", "hidden", h)
      ...

      # access intermediates
      h = self.interms["hidden"])

      # The above is equivalent to:
      # self.get_variable("intermediates", "hidden")[-1]
      return out
  ```

  The interms property can only be used within `.init` / `.apply`.

  Returns:
    The `interms` property
  """

  @property
  def _interms_prop(self: nn.Module):
    return _IntermsAccessor(module=self)

  return _interms_prop


class _IntermsAccessor:
  """Helper object that forwards item access from the interms property."""

  def __init__(
      self,
      module: nn.Module,
  ):
    if module.scope is None:
      raise ValueError(
          '`"interms` property can only be used from within `.init` / `.apply`.'
      )
    self.module = module

  def __getitem__(self, key):
    return self.module.get_variable('intermediates', key)[-1]

  def __setitem__(self, key, value):
    self.module.sow('intermediates', key, value)
