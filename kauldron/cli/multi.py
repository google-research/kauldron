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

"""Multi-command execution CLI."""

from __future__ import annotations

import dataclasses
from typing import Union

from kauldron.cli import cmd_utils as cu
from kauldron.cli import data
from kauldron.cli import inspect_cli
from kauldron.cli import run

_COMMANDS = {
    "data": {
        "element_spec": data.ElementSpec,
        "batch": data.Batch,
    },
    "run": {
        "eval_shape": run.EvalShape,
        "train": run.Train,
        "eval": run.Eval,
    },
    "inspect": {
        "model_overview": inspect_cli.ModelOverview,
    }
}

@dataclasses.dataclass(frozen=True, kw_only=True)
class Execute(cu.SubCommand):
  """Execute multiple commands sequentially."""

  cmds: str = ""

  def __call__(self):
    self.print_config_origin()
    trainer = None

    cmds_list = self.cmds.split(",")
    if not cmds_list or (len(cmds_list) == 1 and not cmds_list[0].strip()):
      print(
          'No subcommands provided to multi. Use --multi.execute.cmds="data'
          ' batch, run train"'
      )
      return

    for cmd_str in cmds_list:
      cmd_str = cmd_str.strip()
      if not cmd_str:
        continue
      parts = cmd_str.split()
      if len(parts) != 2:
        print(f"Invalid command format: {cmd_str}. Expected 'group subcommand'")
        continue
      group_name, sub_name = parts

      if group_name not in _COMMANDS or sub_name not in _COMMANDS[group_name]:
        print(
            f"Unknown command: '{cmd_str}'. Available commands: "
            + ", ".join(
                [f"{g} {s}" for g, subs in _COMMANDS.items() for s in subs]
            )
        )
        continue

      cmd_cls = _COMMANDS[group_name][sub_name]

      print(f"\\n{'='*40}\\nExecuting: {cmd_str}\\n{'='*40}")

      cmd = cmd_cls(cfg=self.cfg, origin=None)  # pytype: disable=wrong-keyword-args

      # Share resolved trainer to avoid redundant recompilations/resolutions
      if trainer is None:
        trainer = cmd.trainer
      else:
        cmd.__dict__["trainer"] = trainer

      cmd()

@dataclasses.dataclass(frozen=True, kw_only=True)
class Multi(cu.CommandGroup):
  """Multi commands."""

  sub_command: Union[Execute]
