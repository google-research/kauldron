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
from typing import Any

from kauldron import konfig
from kauldron.cli import patch_config


@dataclasses.dataclass(frozen=True, kw_only=True)
class CommandGroup(abc.ABC):
  """Base class for all commands."""

  sub_command: SubCommand

  @abc.abstractmethod
  def __call__(
      self,
      cfg: konfig.ConfigDict,
      origin: patch_config.ConfigOrigin | None = None,
  ) -> None:
    pass


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubCommand(abc.ABC):
  """Base class for all sub-commands.

  Takes care of getting and modifying the config, and if needed resolving it.
  This ensures that override and resolution behavior are consistent across
  commands.
  """

  @abc.abstractmethod
  def __call__(
      self,
      cfg: konfig.ConfigDict,
      origin: patch_config.ConfigOrigin | None = None,
  ) -> Any:
    pass
