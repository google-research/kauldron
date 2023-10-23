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

The sweep implementation is structured in 3 classes:

* `SweepsInfo`: Global class containing info on all the sweeps
* `_SweepFnInfo`: Representing a single sweep function (`sweep()`, ...). Each
  sweep function can generate multiple sweep kwargs (that can be combined with
  other sweep functions)
* `SweepInfo`: Representing a single sweep (== work unit).

The XM <> Work unit communication is done through a `--sweep_config` flag that
contain the serialized json kwargs to overwrite.
"""

from __future__ import annotations

# Note: This module is imported by by XM launcher, so should not have any
# extra dependency

import dataclasses
import functools
import itertools
import json
import operator
import types
from typing import Any, Callable, Iterable

from absl import flags
from kauldron import konfig
from kauldron.utils import utils
import ml_collections

_FLAG_NAME = "sweep_config"

_SweepKwargs = dict[str, Any]


@dataclasses.dataclass(kw_only=True)
class SweepsInfo:
  """Container for all sweeps info."""

  sweep_fns: list[_SweepFnInfo]

  @classmethod
  def make(
      cls, *, config_module: types.ModuleType, sweep_names: list[str]
  ) -> SweepsInfo:
    return cls(
        sweep_fns=[
            _SweepFnInfo(module=config_module, sweep_name=sweep_name)
            for sweep_name in sweep_names
        ],
    )

  @functools.cached_property
  def tags(self) -> list[str]:
    return [fn.tag for fn in self.sweep_fns]

  @functools.cached_property
  def all_sweep_info(self) -> list[SweepInfo]:
    """Iterate over the individual sweep."""
    sweeps: list[Iterable[_SweepKwargs]] = []
    for fn_info in self.sweep_fns:
      sweeps.append(fn_info.fn())

    all_sweep_info = []
    for sweep_kwargs in itertools.product(*sweeps):
      sweep_kwargs = functools.reduce(operator.ior, sweep_kwargs, {})
      all_sweep_info.append(SweepInfo(sweep_kwargs=sweep_kwargs))
    return all_sweep_info


@dataclasses.dataclass(kw_only=True)
class _SweepFnInfo:
  """Info for a single sweep function."""

  module: types.ModuleType
  sweep_name: str

  @functools.cached_property
  def tag(self) -> str:
    """Xmanager tag."""
    return "🧹" if not self.sweep_name else f"🧹{self.sweep_name}"

  @functools.cached_property
  def fn_name(self) -> str:
    return "sweep" if not self.sweep_name else f"sweep_{self.sweep_name}"

  @property
  def fn(self) -> Callable[[], Iterable[_SweepKwargs]]:
    """Fetch the sweep function."""
    fn = getattr(self.module, self.fn_name, None)
    if fn is None:
      raise ValueError(f"Could not find sweep function '{self.fn_name}()'")
    return fn


@dataclasses.dataclass(kw_only=True)
class SweepInfo:
  """Info for a single sweep (== work unit)."""

  sweep_kwargs: _SweepKwargs

  @functools.cached_property
  def xm_ui_kwargs(self) -> _SweepKwargs:
    """Arguments to be displayed in the XM UI."""
    # TODO(epot): If str is too big, should truncate.
    return {k: repr(v) for k, v in self.sweep_kwargs.items()}

  @functools.cached_property
  def xm_flags(self) -> dict[str, str]:
    """Kwargs to be passed to the work unit binary."""
    # Use custom encoder to support ConfigDict objects
    return {_FLAG_NAME: JsonEncoder().encode(self.sweep_kwargs)}


class JsonEncoder(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, konfig.ConfigDict):
      return json.loads(o.to_json())
    else:
      return super().default(o)


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


def _assert_is_config(cfg):
  if not isinstance(cfg, (list, dict, konfig.ConfigDict)):
    raise TypeError("Error for field {}: Sweep update expect the config")


def update_with_sweep(
    config: konfig.ConfigDict,
    sweep_kwargs: str,
) -> konfig.ConfigDict:
  """Update the config with sweep."""
  # Might create issue with adhoc import, but `update_with_sweep` is likely not
  # called on Colab
  from kauldron.utils import paths  # pylint: disable=g-import-not-at-top

  if not sweep_kwargs:
    return config

  # Could support more fancy flags overwrite (e.g. `model.*.dtype = `)
  assert isinstance(config, (dict, konfig.ConfigDict))

  sweep_kwargs = json.loads(sweep_kwargs)
  # Normalize to tuple
  sweep_kwargs: dict[str, Any] = utils.json_list_to_tuple(sweep_kwargs)

  for k, v in sweep_kwargs.items():
    root = config

    *parts, target = paths.Path.from_str(k).parts
    for part in parts:
      root = root[part]
      if not isinstance(root, (list, dict, ml_collections.ConfigDict)):
        raise TypeError(
            f"Cannot overwrite sweep arg {k}: {part} is unsuported type"
            f" {type(root)}. Please open an issue if this should be fixed."
        )

    root[target] = v

  return config


def _normalize_key(obj, k):
  if isinstance(obj, list) and k.isdigit():
    k = int(k)
  return k
