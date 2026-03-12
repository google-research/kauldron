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
from typing import Any

from etils import epy
from kauldron import konfig
from kauldron.cli import patch_config


@dataclasses.dataclass(frozen=True, kw_only=True)
class CommandGroup:
  """Base class for all commands.

  These are only containers to group sub-commands in the CLI and can be used to
  define shared flags (e.g. --datasets, --eval_mode).

  These flags are automatically registered by simple_parsing from the
  attributes of the class.

  Subclasses of this class need to override the `sub_command` attribute, which
  and annotate it with a Union of the desired sub-commands to make them
  available as positional CLI commands.
  """

  sub_command: SubCommand


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubCommand(abc.ABC):
  """Base class for all sub-commands.

  Subclasses of this class need to be dataclasses and implement `__call__`.

  Additional CLI flags can be defined as attributes and will be automatically
  be picked up and registered by simple_parsing.
  """

  # `cmd=False` prevents simple_parsing from registering --cfg as a CLI arg,
  # which would conflict with the absl --cfg flag defined in main.py.
  cfg: konfig.ConfigDict | None = dataclasses.field(
      default=None, metadata={"cmd": False}
  )
  origin: patch_config.ConfigOrigin | None = dataclasses.field(
      default=None, metadata={"cmd": False}
  )
  group: CommandGroup | None = dataclasses.field(
      default=None, metadata={"cmd": False}
  )

  @functools.cached_property
  def trainer(self):
    return konfig.resolve(self.cfg)

  @abc.abstractmethod
  def __call__(self) -> Any:
    """Executes the command and return a printable object."""


def execute_command(
    group: CommandGroup,
    cfg: konfig.ConfigDict,
    origin: patch_config.ConfigOrigin,
) -> Any:
  """Executes the command and pretty-print the result."""
  cmd = dataclasses.replace(
      group.sub_command, cfg=cfg, origin=origin, group=group
  )
  output = cmd()
  epy.pprint(output)
