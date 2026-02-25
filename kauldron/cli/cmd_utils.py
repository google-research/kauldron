# Copyright 2026 The kauldron Authors.
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

"""Base class for all sub-commands."""

from __future__ import annotations

import abc
import dataclasses
import functools

from kauldron import konfig


@dataclasses.dataclass(frozen=True)
class SubCommand(abc.ABC):
  """Base class for all sub-commands.

  Takes care of getting and modifying the config, and if needed resolving it.
  This ensures that override and resolution behavior are consistent across
  commands.
  """

  # `cmd=False` prevents simple_parsing from registering --cfg as a CLI arg,
  # which would conflict with the absl --cfg flag defined in main.py.
  cfg: konfig.ConfigDict | None = dataclasses.field(
      default=None, metadata={"cmd": False}
  )

  @functools.cached_property
  def trainer(self):
    return konfig.resolve(self.cfg)

  # TODO(klausg): should this be __call__ instead?
  @abc.abstractmethod
  def execute(self) -> str:
    pass
