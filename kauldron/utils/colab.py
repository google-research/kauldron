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

"""Colab utils."""

import enum
import types

from etils import epy
from kauldron import konfig
from kauldron import kontext


with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  from etils import ecolab
  from kauldron.xm._src import kauldron_utils
  from kauldron.xm._src import sweep_cfg_utils

  from colabtools import interactive_forms
  # pylint: enable=g-import-not-at-top  # pytype: enable=import-error


class SweepMode(epy.StrEnum):
  """Sweep mode."""

  NONE = enum.auto()
  FIRST = enum.auto()
  ALL = 'ALL (resolve all, train last)'


def iter_sweep_configs(
    *,
    module: types.ModuleType,
    sweep_mode: str,
    sweep_name: str,
    config_args: str | None = None,
):
  """Iterates over sweep configs."""
  sweep_mode = SweepMode(sweep_mode)
  assert sweep_mode != SweepMode.NONE

  _update_sweep_names_forms(module)

  # Select the sweeps to resolve.
  sweep_info = kauldron_utils.KauldronSweep(
      _module=module,
      _sweep_value=sweep_name,
  )  # pylint: disable=unexpected-keyword-arg
  all_sweep_items = list(sweep_info)
  if sweep_mode == SweepMode.FIRST:
    all_sweep_items = all_sweep_items[:1]

  output_cfg = []
  with ecolab.collapse(f'Resolving {len(all_sweep_items)} sweeps configs'):
    for i, sweep_item in enumerate(all_sweep_items):
      # Re-create the config to avoid mutations leak between iterations.
      if config_args:
        cfg = module.get_config(config_args)
      else:
        cfg = module.get_config()

      sweep_kwargs = kauldron_utils.deserialize_job_kwargs(
          sweep_item.job_kwargs
      )
      # TODO(epot): Display the sweep short name (workdir) and config.
      for k, v in sweep_kwargs.items():
        kontext.set_by_path(cfg, k, v)
      # Only for visualization.
      sweep_cfg_overwrites = konfig.ConfigDict(sweep_kwargs)
      print(f'Work-unit {i+1}:', flush=True)
      # TODO(epot): Should report IPython.display.HTML bug to Colab team.
      # Use `print` rather than `ecolab.disp` as HTML display bug here for
      # some reason and mangle the previous outputs.
      print(sweep_cfg_overwrites)
      print()

      # Somhow, yield here remove some of the outputs, so instead append and
      # returns everything at the end.
      output_cfg.append(cfg)

    return output_cfg


def _update_sweep_names_forms(module: types.ModuleType) -> None:
  """Updates the `SWEEP_NAMES` field in Colab."""
  all_names = sweep_cfg_utils.all_available_sweep_names(module)
  interactive_forms.UpdateParam(
      'SWEEP_NAMES',
      config=['*'] + list(all_names),
      update_globally=True,
  )
