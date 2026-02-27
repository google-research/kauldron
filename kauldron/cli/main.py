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
import tempfile

from absl import app
from absl import flags
import jax
from kauldron import konfig
from kauldron.cli import cmd_utils
from kauldron.cli import config
from kauldron.cli import data
from kauldron.cli import patch_config
import simple_parsing

FLAGS = flags.FLAGS

_CONFIG = konfig.DEFINE_config_file(
    "cfg",
    None,
    "Path to the Kauldron configuration file.",
    lock_config=False,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Args:
  """Kauldron CLI for inspecting and debugging configurations."""

  command: config.Config | data.Data

  patch: patch_config.PatchConfig = dataclasses.field(
      default_factory=patch_config.PatchConfig,
      # Manually added in flag_parser() to support "--patch." prefix.
      metadata={"cmd": False},
  )


def flag_parser(argv: list[str]) -> Args:
  """Parses CLI flags from argv using simple_parsing."""
  args_parser = simple_parsing.ArgumentParser(
      prog="kauldron",
      description="Kauldron CLI entry point.",
  )
  args_parser.add_arguments(Args, dest="args")
  args_parser.add_arguments(
      patch_config.PatchConfig, dest="patch", prefix="patch."
  )

  with konfig.set_lazy_imported_modules():
    # TODO(klausg): why argv[1:]?
    namespace, remaining_argv = args_parser.parse_known_args(argv[1:])

    FLAGS([""] + remaining_argv)

  return namespace.args


def _get_config(
    patcher: patch_config.PatchConfig,
) -> tuple[konfig.ConfigDict, patch_config.ConfigOrigin]:
  """Returns a config and its provenance."""
  cfg = _CONFIG.value
  filename = FLAGS["cfg"].config_filename  # pytype: disable=attribute-error
  overrides = FLAGS["cfg"].override_values  # pytype: disable=attribute-error

  patches = {}
  with tempfile.TemporaryDirectory(prefix="kauldron_") as workdir:
    if cfg.workdir is None:
      cfg.workdir = workdir
    patches["workdir"] = cfg.workdir

  # Apply Patches
  final_cfg = patcher(cfg)  # TODO(klausg): keep track of patch-overrides

  provenance = patch_config.ConfigOrigin(
      filename=filename,
      overrides=overrides,
      patches=patches,
  )

  return final_cfg, provenance


def main(args: Args) -> None:
  """Dispatches to the selected CLI command."""
  patcher: patch_config.PatchConfig = args.patch
  cfg, origin = _get_config(patcher)

  command: cmd_utils.CommandGroup = args.command
  command(cfg=cfg, origin=origin)


if __name__ == "__main__":
  # Adds jax flags to the program.
  jax.config.config_with_absl()
  app.run(main)  # external
