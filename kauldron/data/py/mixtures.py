# Copyright 2025 The kauldron Authors.
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
from kauldron import random
from kauldron.data.py import base


@dataclasses.dataclass(frozen=True)
class Mix(base.PyGrainPipeline):
  """Create a dataset mixture from given weights."""

  datasets: list[base.PyGrainPipeline]

  _: dataclasses.KW_ONLY
  weights: None | list[float | int] = None
  shuffle: bool = True

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    # Ensure each nested dataset gets a different seed.
    datasets = [
        ds.ds_with_transforms(rng.fold_in(i))
        for i, ds in enumerate(self.datasets)
    ]

    ds = grain.MapDataset.mix(datasets, weights=self.weights)

    if self.shuffle:
      # Sometimes, the call to `as_seed()` returns a 64-bit unsigned int. The
      # modulo ensures that the seed fits into a 32-bit unsigned integer which
      # is the expected type by `grain.MapDataset.shuffle`.
      seed = rng.fold_in("shuffle").as_seed() % (2 ** 32)
      ds = ds.shuffle(seed=seed)
    return ds


@dataclasses.dataclass(frozen=True)
class SelectFromDatasets(base.PyGrainPipeline):
  """Create a dataset mixture using a selection map."""

  datasets: list[base.PyGrainPipeline]

  _: dataclasses.KW_ONLY
  selection_map: grain.DatasetSelectionMap
  shuffle: bool = True

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    # Ensure each nested dataset gets a different seed.
    datasets = [
        ds.ds_with_transforms(rng.fold_in(i))
        for i, ds in enumerate(self.datasets)
    ]

    ds = grain.MapDataset.select_from_datasets(
        datasets, selection_map=self.selection_map
    )

    if self.shuffle:
      ds = ds.shuffle(rng.fold_in("shuffle").as_seed())
    return ds
