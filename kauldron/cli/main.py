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
from kauldron.cli import cmd_utils as cu
from kauldron.cli import config
from kauldron.cli import data
from kauldron.cli import patch_config
from kauldron.cli import run
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

  command: config.Config | data.Data | run.Run

  patch: patch_config.PatchConfig = dataclasses.field(
      default_factory=patch_config.PatchConfig,
      # Manually added in flag_parser() to support "--patch." prefix.
      metadata={"cmd": False},
  )


def flag_parser(argv: list[str]) -> Args:
  """Parses CLI flags from argv using simple_parsing."""
  with konfig.set_lazy_imported_modules():
    patch, remaining_argv = _parse_patch_flags(argv[1:])
    args, remaining_argv = _parse_main_args(remaining_argv)

    FLAGS(argv[:1] + remaining_argv)

  return dataclasses.replace(args, patch=patch)


def _parse_patch_flags(
    argv: list[str],
) -> tuple[patch_config.PatchConfig, list[str]]:
  """Parses --patch.* flags and returns (PatchConfig, remaining_argv).

  Uses a dedicated parser without subparsers so argparse can consume
  --patch.* flags regardless of their position in argv.

  Args:
    argv: Command-line arguments without the program name.

  Returns:
    A tuple of (PatchConfig, remaining_argv).
  """
  patch_parser = simple_parsing.ArgumentParser(add_help=False)
  patch_parser.add_arguments(
      patch_config.PatchConfig, dest="patch", prefix="patch."
  )
  namespace, remaining_argv = patch_parser.parse_known_args(argv)
  return namespace.patch, remaining_argv


def _parse_main_args(
    argv: list[str],
) -> tuple[Args, list[str]]:
  """Parses the main Args (command + sub_command) from argv."""
  args_parser = simple_parsing.ArgumentParser(
      prog="kauldron",
      description="Kauldron CLI for inspecting and debugging configurations.",
      epilog=(
          "config flags:\n"
          "  --cfg CFG           Path to the Kauldron config file.\n"
          "  --cfg.KEY=VALUE     Override config values.\n"
      ),
  )
  args_parser.add_arguments(
      patch_config.PatchConfig, dest="_patch", prefix="patch."
  )
  args_parser.add_arguments(Args, dest="args")
  # Hacky way to tidy up the help message:
  # args_parser._preprocessing(args=argv)
  # for group in args_parser._action_groups:
  #   if "PatchConfig" in (group.title or ""):
  #     group.title = "patch flags"
  #   if group.title and group.title.startswith("Args"):
  #     group.title = None
  #     group.description = None
  namespace, remaining_argv = args_parser.parse_known_args(argv)
  return namespace.args, remaining_argv


def main(args: Args) -> None:
  """Dispatches to the selected CLI command."""
  # Get relevant info from commandline arguments
  patcher: patch_config.PatchConfig = args.patch
  cfg = _CONFIG.value
  filename = FLAGS["cfg"].config_filename  # pytype: disable=attribute-error
  overrides = FLAGS["cfg"].override_values  # pytype: disable=attribute-error

  # Workdir management
  with tempfile.TemporaryDirectory(prefix="kauldron_") as workdir:
    if cfg.workdir is None:
      overrides |= cu.tracked_update(cfg, "workdir", workdir)

    patched_config, patches = patcher(cfg)

    conflicts = set(overrides) & set(patches)
    for key in conflicts:
      cu.tracked_update(patched_config, key, overrides[key])
      del patches[key]

    origin = patch_config.ConfigOrigin(
        filename=filename,
        overrides=overrides,
        patches=patches,
    )

    cu.execute_command(
        group=args.command,
        cfg=patched_config,
        origin=origin,
    )


if __name__ == "__main__":
  # Adds jax flags to the program.
  jax.config.config_with_absl()
  app.run(main)  # external
