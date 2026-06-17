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

"""Dataset mixtures."""

from __future__ import annotations

import dataclasses

import grain.python as grain
import jax
from kauldron import random
from kauldron.data.py import base
from kauldron.ktyping import PRNGKey


@dataclasses.dataclass(frozen=True)
class Mix(base.PyGrainPipeline):
  """Create a dataset mixture from given weights."""

  datasets: list[base.PyGrainPipeline]

  _: dataclasses.KW_ONLY
  weights: None | list[float | int] = None
  shuffle: bool = True

  def ds_for_current_process(self, rng: PRNGKey) -> grain.MapDataset:
    # Ensure each nested dataset gets a different seed.
    datasets = [
        ds.ds_with_transforms(jax.random.fold_in(rng, i))
        for i, ds in enumerate(self.datasets)
    ]

    ds = grain.MapDataset.mix(datasets, weights=self.weights)

    if self.shuffle:
      seed = random.random_seed(random.fold_in_str(rng, "shuffle"))
      ds = ds.shuffle(seed=seed)
    return ds


@dataclasses.dataclass(frozen=True)
class SelectFromDatasets(base.PyGrainPipeline):
  """Create a dataset mixture using a selection map."""

  datasets: list[base.PyGrainPipeline]

  _: dataclasses.KW_ONLY
  selection_map: grain.DatasetSelectionMap
  shuffle: bool = True

  def ds_for_current_process(self, rng: PRNGKey) -> grain.MapDataset:
    # Ensure each nested dataset gets a different seed.
    datasets = [
        ds.ds_with_transforms(jax.random.fold_in(rng, i))
        for i, ds in enumerate(self.datasets)
    ]

    ds = grain.MapDataset.select_from_datasets(
        datasets, selection_map=self.selection_map
    )

    if self.shuffle:
      ds = ds.shuffle(random.random_seed(random.fold_in_str(rng, "shuffle")))
    return ds
