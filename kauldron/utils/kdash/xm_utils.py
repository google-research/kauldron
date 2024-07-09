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

"""XM utils."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

from etils import exm
import flax


def get_sweep_argnames(xp: xmanager_api.Experiment) -> list[str]:
  """Return a mapping sweep-keys to a set of their values for given XID.

  Sweep-keys are all keys that are set in any of the work-units, and which do
  not have a constant value.
  Args:
    xp: The xmanager experiment to get the sweep info for.

  Returns:
    A dictionary with each sweep-key and the set of corresponding values that
    key takes among any work unit.
  """
  # raw sweep is just a list of config overrides for all work-units
  wu_params = [wu.parameters for wu in xp.get_work_units()]
  num_work_units = len(wu_params)
  # merge them by keys into a single tree with a list of possible values
  all_override_keys = {k for wup in wu_params for k in wup}  # pylint: disable=g-complex-comprehension
  zipped_sweep_values = {
      k: [_freeze(params[k]) for params in wu_params if k in params]
      for k in all_override_keys
  }
  # filter out simple overrides (non-sweep keys), which are characterized by:
  # 1. they occur for every wu and 2. they always have the same value
  filtered_sweep_set = {
      k: set(v)
      for k, v in zipped_sweep_values.items()
      if len(v) < num_work_units or len(set(v)) > 1
  }
  return list(filtered_sweep_set)


def _freeze(v: Any) -> Hashable:
  # convert values to frozen so they become hashable
  match v:
    case list() as l:
      return tuple(l)
    case dict() as d:
      return flax.core.FrozenDict(d)
    case set() as s:
      return frozenset(s)
    case int() | float() | bool() | bytes() | None as basic:
      return basic
    case _ as other:
      return str(other)


def add_flatboard_artifact(name: str, url: str) -> None:
  xp = exm.current_experiment()

  # Already exists, do not create it to avoid duplicates between work-units.
  for old_artifact in xp.get_artifacts(
      artifact_types=[xmanager_api.ArtifactType.ARTIFACT_TYPE_FLATBOARD_URL]
  ):
    if old_artifact.description == name:
      return

  xp.create_artifact(
      artifact_type=xmanager_api.ArtifactType.ARTIFACT_TYPE_FLATBOARD_URL,
      artifact=url,
      description=name,
  )
