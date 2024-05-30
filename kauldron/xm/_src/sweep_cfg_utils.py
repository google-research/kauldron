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

"""Sweep from the `def sweep()` from the `--cfg` config file."""

from __future__ import annotations

from collections.abc import Iterable
import dataclasses
import functools
import itertools
import operator
import types
from typing import Self

from kauldron import konfig
from kauldron.xm._src import cfg_provider_utils
from kauldron.xm._src import jobs_info
from kauldron.xm._src import sweep_utils


@dataclasses.dataclass(frozen=True, kw_only=True)
class SweepFromCfg(sweep_utils.SweepInfo):
  """Sweep from the `--cfg` config file.

  Run the named sweeps defined by `sweep_[NAME]()` in the config file.
  If multiple sweep names are given run all their combinations (product).
  Empty string match `def sweep()` (default).

  Example:

  ```
  def sweep():
    for batch_size in [16, 32, 64]:
      yield {"cfg.batch_size": batch_size}
  ```
  """
  # Module from which to extract the sweep functions, automatically set inside
  # `replace_with_job_provider`
  _module: types.ModuleType = dataclasses.field(  # pytype: disable=annotation-type-mismatch
      default=None,
      repr=False,
  )

  def __iter__(self) -> Iterable[sweep_utils.SweepItem]:
    if self._module is None:
      raise ValueError(
          f"{type(self).__name__}: sweep module not set. Did you set"
          " `--cfg=path/to/config.py` ?"
      )
    yield from _sweeps_from_module(
        module=self._module,  # pylint: disable=attribute-error
        names=self.sweep_names,
    )

  @functools.cached_property
  def sweep_names(self) -> list[str]:
    match self._sweep_value:
      case None | False:
        return []
      case True:
        return [""]
      case "*":  # All `sweep_` functions
        return all_available_sweep_names(self._module)
      case str():
        return self._sweep_value.split(",")
      case list():
        return self._sweep_value
      case _:
        raise ValueError(f"Unexpected sweep value: {self._sweep_value}")

  @functools.cached_property
  def tags(self) -> list[str]:
    return [f"🧹{name}" for name in self.sweep_names]

  def replace_with_jobs_provider(
      self, jobs_provider: jobs_info.JobsProvider
  ) -> Self:
    if isinstance(
        jobs_provider.cfg_provider, cfg_provider_utils.EmptyConfigProvider
    ):
      # `cfg_provider` can be set later in `launch.py` (as it is defined by the
      # `--cfg` flag).
      return self
    return dataclasses.replace(self, _module=jobs_provider.cfg_provider.module)


def _sweeps_from_module(
    module: types.ModuleType, names: list[str]
) -> Iterable[sweep_utils.SweepItem]:
  # Step 1: Collect all sweep functions
  sweeps = [_get_sweep_fn(module, name) for name in names]

  # Step 2: Merge all sweep functions with product
  for sweep_kwargs in itertools.product(*sweeps):
    sweep_kwargs = functools.reduce(operator.ior, sweep_kwargs, {})
    yield sweep_utils.SweepItem(job_kwargs=sweep_kwargs)


def _ui_repr(v):
  """Parameters displayed on the UI."""
  # TODO(epot): In theory, could list exhaustivelly all accepted types
  if isinstance(v, (bool, int, str, float, type(None))):
    return v
  repr_ = repr(v)
  if isinstance(v, konfig.ConfigDict):
    repr_ = repr_.removeprefix("<ConfigDict[").removesuffix("]>")
  # TODO(epot): If str is too big, should truncate ?
  return repr_


def _get_sweep_fn(module: types.ModuleType, fn_name: str):
  fn_name = "sweep_" + fn_name if fn_name else "sweep"
  fn = getattr(module, fn_name, None)
  if fn is None:
    available_sweeps = all_available_sweep_names(module)
    raise ValueError(
        f"Could not find sweep function '{fn_name}()' in {module}."
        f" Available sweeps: {available_sweeps}"
    )
  return fn()


def all_available_sweep_names(module: types.ModuleType) -> list[str]:
  """Returns all available sweep names."""
  sweep_names = []
  for name in dir(module):
    if name.startswith("sweep_"):
      sweep_names.append(name.removeprefix("sweep_"))
    elif name == "sweep":
      sweep_names.append("")
  return sweep_names
