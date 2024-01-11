# Copyright 2023 The kauldron Authors.
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

"""Sweep utils.

The XM <> Work unit communication is done through a `--sweep_config` flag that
contain the serialized json kwargs to overwrite.
"""

from __future__ import annotations

import json
from typing import Any

from absl import flags
from kauldron import konfig
from kauldron.utils import utils
import ml_collections

# This should match sweep args passed in `kauldron/xm/_src/kauldron_utils.py`
_FLAG_NAME = "sweep_config"


def define_sweep_flag() -> flags.FlagHolder[str | None]:
  """Binary flag to load the sweep info.

  Usage:

  ```
  _CONFIG = config_flags.DEFINE_config_file('cfg', ..., lock_config=False)
  _SWEEP_CONFIG = sweep_utils.define_sweep_flag()


  def main(_):
    # First update the config with the sweep values
    cfg = sweep_utils.update_with_sweep(_CONFIG.value, _SWEEP_CONFIG.value)

    # Resolve the config after updates
    cfg = konfig.resolve(cfg)
  ```

  Returns:
    The flag holder value
  """
  return flags.DEFINE_string(_FLAG_NAME, None, "Training configuration.")


def update_with_sweep(
    config: konfig.ConfigDict,
    sweep_kwargs: str,
) -> konfig.ConfigDict:
  """Update the config with sweep."""
  # Might create issue with adhoc import, but `update_with_sweep` is likely not
  # called on Colab
  from kauldron import kontext  # pylint: disable=g-import-not-at-top

  if not sweep_kwargs:
    return config

  # Could support more fancy flags overwrite (e.g. `model.*.dtype = `)
  assert isinstance(config, (dict, konfig.ConfigDict))

  sweep_kwargs = json.loads(sweep_kwargs)
  # Normalize to tuple
  sweep_kwargs: dict[str, Any] = utils.json_list_to_tuple(sweep_kwargs)

  for k, v in sweep_kwargs.items():
    root = config

    *parts, target = kontext.Path.from_str(k).parts
    for part in parts:
      root = root[part]
      if not isinstance(root, (list, dict, ml_collections.ConfigDict)):
        raise TypeError(
            f"Cannot overwrite sweep arg {k}: {part} is unsuported type"
            f" {type(root)}. Please open an issue if this should be fixed."
        )

    root[target] = v

  return config
