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

"""XManager laucher script for the kauldron codebase.

This is a thin wrapper around `kxm.Experiment` which allow to overwrite
`Experiment` attributes through CLI.

See instructions at.
"""

from __future__ import annotations

from absl import app
from absl import flags
from etils import epy
# Import jax outside the `binary_adhoc` scope to avoid some errors
import jax  # pylint: disable=unused-import

# Re-import `epy` from HEAD (as the XManager CLI might contain an old version)
with epy.binary_adhoc(reload="etils"):
  from etils import epy  # pylint: disable=g-import-not-at-top,reimported


with epy.binary_adhoc():
  # pylint: disable=g-import-not-at-top
  from kauldron import kxm
  from kauldron import konfig  # pylint: disable=unused-import,g-import-not-at-top
  # pylint: enable=g-import-not-at-top

# Required for TQDM
flags.FLAGS.xm_progress_reporting_use_notes = True


try:
  _XP = konfig.DEFINE_config_file(
      "xp",
      "third_party/py/kauldron/xm/configs/kd_base.py",
      "Path to the XManager config to be run.",
      lock_config=False,
  )
except Exception as e_:  # pylint: disable=broad-exception-caught
  epy.reraise(e_, suffix="See all flags at")

_CONFIG = konfig.DEFINE_config_file(
    "cfg",
    None,
    "Path to the configuration file to be run.",
    lock_config=False,
)


def main(_) -> None:
  """Main launcher code."""
  xp_config = _XP.value
  job_config = _CONFIG.value

  # Execute resolve within a adhoc import context as it can import
  # additional modules
  with epy.binary_adhoc():
    try:
      xp: kxm.Experiment = konfig.resolve(xp_config)
    except TypeError as e:
      epy.reraise(e, suffix="See all flags at")

  # Merge with the `--cfg` flags
  if job_config is not None:
    with (
        epy.binary_adhoc(),
        konfig.set_lazy_imported_modules(),
    ):
      cfg_provider = kxm.ConfigProvider.from_flag(_CONFIG)
    # Merge the `XManager` with the config flag provider
    xp = xp.replace(cfg_provider=cfg_provider)

  xp.launch()


def _flags_parser(args: list[str]) -> None:
  """Flag parser."""
  # Lazy-import everything (TF,... not included in the XM launcher)
  # The kxm, XM,... are resolved in `konfig.resolve`
  # Could pottentially exclude `xmanager`, `kxm`,...
  with (
      konfig.set_lazy_imported_modules(),
      epy.binary_adhoc(),
  ):
    try:
      flags.FLAGS(args)
    except Exception as e:  # pylint: disable=broad-exception-caught
      epy.reraise(e, suffix="See all flags at")


if __name__ == "__main__":
  app.run(main, flags_parser=_flags_parser)
