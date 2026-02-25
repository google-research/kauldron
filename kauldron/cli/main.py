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

r"""Kauldron CLI entry point.

Usage:
    kauldron <command> <sub_command> --cfg=<path/to/config.py>
        [--cfg.override.key=value ...]
"""

from __future__ import annotations

import dataclasses

from absl import app
from absl import flags
from etils import eapp
from kauldron import konfig
from kauldron import kontext
from kauldron.cli import cmd_utils
from kauldron.cli import config
from kauldron.cli import data
import simple_parsing

FLAGS = flags.FLAGS

_CONFIG = konfig.DEFINE_config_file(
    "cfg",
    None,
    "Path to the Kauldron configuration file.",
    lock_config=False,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class MutationArgs:
  """Arguments for mutating the config."""

  # Set the cfg.workdir (required to run). Defaults to /tmp/kauldron.
  workdir: str | None = "/tmp/kauldron"

  # Stop training after N steps. Defaults to 1.
  stop_after_steps: int | None = 1

  # Override the batch size in train and eval datasets.
  batch_size: int | None = None

  # Whether to run eval. Defaults to True.
  eval: bool = True

  # Whether to run checkpointer. Defaults to False.
  checkpointer: bool = False

  # Whether to compute metrics. Defaults to True.
  metrics: bool = True

  # Whether to compute summaries. Defaults to True.
  summaries: bool = True

  def mutate_config(self, cfg: konfig.ConfigDict) -> konfig.ConfigDict:
    """Returns a mutated config."""
    if self.stop_after_steps is not None:
      cfg.stop_after_steps = self.stop_after_steps

    if self.batch_size is not None:
      kontext.set_by_path(cfg, "train_ds.**.batch_size", self.batch_size)
      kontext.set_by_path(cfg, "train_ds.**.shuffle_buffer_size", 1)
      if hasattr(cfg, "evals"):
        kontext.set_by_path(cfg, "evals.**.batch_size", self.batch_size)
        kontext.set_by_path(cfg, "evals.**.num_batches", 1)

    if not self.eval:
      cfg.evals = {}

    if not self.checkpointer:
      cfg.checkpointer = None

    if not self.metrics:
      cfg.train_metrics = {}

    if not self.summaries:
      cfg.train_summaries = {}

    return cfg


@dataclasses.dataclass(frozen=True, kw_only=True)
class Args:
  """Kauldron CLI for inspecting and debugging configurations."""

  mutation_args: MutationArgs

  command: config.ConfigCmd | data.DataCmd = simple_parsing.subparsers({
      "config": config.ConfigCmd,
      "data": data.DataCmd,
  })


flag_parser = eapp.make_flags_parser(Args)


def main(args: Args) -> None:
  """Dispatches to the selected CLI command."""
  if not hasattr(args, "command"):
    raise SystemExit("No command specified. Use --help for usage.")
  cmd = args.command
  sub_cmd = cmd.sub_command
  assert isinstance(sub_cmd, cmd_utils.SubCommand)
  cfg = _CONFIG.value
  updated_cfg = args.mutation_args.mutate_config(cfg)

  sub_cmd = dataclasses.replace(sub_cmd, cfg=updated_cfg)
  cmd = dataclasses.replace(cmd, sub_command=sub_cmd)
  cmd.execute()


if __name__ == "__main__":
  with konfig.set_lazy_imported_modules():
    app.run(main, flags_parser=flag_parser)
