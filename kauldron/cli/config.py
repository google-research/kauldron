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

"""Config-related CLI commands."""

from __future__ import annotations

import dataclasses

from etils import epy
from kauldron.cli import cmd_utils as cu


@dataclasses.dataclass(frozen=True, kw_only=True)
class Show(cu.SubCommand):
  """Display the unresolved config tree."""

  def __call__(self):
    self.print_config_origin()
    epy.pprint(self.cfg)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Resolve(cu.SubCommand):
  """Resolve and display the fully-instantiated config."""

  verbose: bool = True

  def __call__(self):
    self.print_config_origin()
    trainer = self.trainer  # trigger config resolution
    if self.verbose:
      print("======== Resolved Config ========")
      epy.pprint(trainer)

    print("Successfully resolved config!")


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config(cu.CommandGroup):
  """Config command group."""

  sub_command: Show | Resolve
