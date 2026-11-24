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

"""XManager launcher script for the kauldron codebase.

This is a thin wrapper around `kxm.Experiment` which allows overwriting
`Experiment` attributes through CLI.

See instructions at.
"""

from __future__ import annotations

from typing import Any

from absl import app
from absl import flags
from etils import epy

# Re-import `epy` from HEAD (as the XManager CLI might contain an old version)
with epy.binary_adhoc(reload="etils"):
  from etils import epy  # pylint: disable=g-import-not-at-top,reimported

with epy.binary_adhoc():
  # pylint: disable=g-import-not-at-top
  from kauldron import konfig
  from kauldron import kxm
  # pylint: enable=g-import-not-at-top

# Required for TQDM
flags.FLAGS.xm_progress_reporting_use_notes = True

_DEFAULT_XP_CONFIG = "third_party/py/kauldron/xm/configs/kd_base.py"


def define_flags(
    flag_values: flags.FlagValues = flags.FLAGS,
) -> tuple[flags.FlagHolder[Any], flags.FlagHolder[Any]]:
  """Defines `--xp` and `--cfg` launcher flags on `flag_values`."""
  with (
      epy.binary_adhoc(),
      konfig.set_lazy_imported_modules(),
  ):
    try:
      xp_flag = konfig.DEFINE_config_file(
          "xp",
          _DEFAULT_XP_CONFIG,
          "Path to the XManager config to be run.",
          flag_values=flag_values,
          lock_config=False,
      )
    except Exception as e_:  # pylint: disable=broad-exception-caught
      epy.reraise(e_, suffix="See all flags at")

    cfg_flag = konfig.DEFINE_config_file(
        "cfg",
        None,
        "Path to the configuration file to be run.",
        flag_values=flag_values,
        lock_config=False,
    )
  return xp_flag, cfg_flag


def parse_flags(
    args: list[str],
    *,
    flag_values: flags.FlagValues = flags.FLAGS,
) -> list[str]:
  """Parses CLI flags while keeping `--cfg`/`--xp` imports lazy."""
  # Lazy-import everything (TF, JAX, ... not included in the XM launcher).
  # The kxm, XM, ... are resolved in `konfig.resolve`.
  with (
      konfig.set_lazy_imported_modules(),
      epy.binary_adhoc(),
  ):
    try:
      return flag_values(args)
    except Exception as e:  # pylint: disable=broad-exception-caught
      epy.reraise(e, suffix="See all flags at")


def launch_from_flags(
    xp_flag: flags.FlagHolder[Any],
    cfg_flag: flags.FlagHolder[Any],
) -> None:
  """Resolves launcher flags and launches the XManager experiment."""
  with (
      epy.binary_adhoc(),
      konfig.set_lazy_imported_modules(),
  ):
    xp_config = xp_flag.value
    job_config = cfg_flag.value

  # Execute resolve within an adhoc import context as it can import
  # additional modules.
  with epy.binary_adhoc():
    try:
      xp: kxm.Experiment = konfig.resolve(xp_config)  # pyrefly: ignore[bad-assignment]
    except TypeError as e:
      epy.reraise(e, suffix="See all flags at")

  # Merge with the `--cfg` flags.
  if job_config is not None:
    with (
        epy.binary_adhoc(),
        konfig.set_lazy_imported_modules(),
    ):
      cfg_provider = kxm.ConfigProvider.from_flag(cfg_flag, config=job_config)
    xp = xp.replace(cfg_provider=cfg_provider)
  elif cfg_flag.present:
    raise ValueError(
        f"--{cfg_flag.name} was provided, but `get_config()` returned None."
    )

  xp.launch()


_XP, _CONFIG = define_flags()


def main(_) -> None:
  """Main launcher code."""
  launch_from_flags(_XP, _CONFIG)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_flags)
