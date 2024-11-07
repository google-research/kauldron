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

"""Dataset mixtures."""

from __future__ import annotations

import dataclasses

import grain.python as grain
from kauldron import random
from kauldron.data.py import base


@dataclasses.dataclass(frozen=True)
class Mix(base.PyGrainPipeline):
  """Create a dataset mixture."""

  datasets: list[base.PyGrainPipeline]

  _: dataclasses.KW_ONLY
  weights: None | list[float | int] = None
  shuffle: bool = True

  def ds_for_current_process(self, rng: random.PRNGKey) -> grain.MapDataset:
    # We make sure each nested dataset get a different seed
    datasets = [
        ds.ds_with_transforms(rng.fold_in(i))
        for i, ds in enumerate(self.datasets)
    ]

    ds = grain.MapDataset.mix(datasets, weights=self.weights)

    if self.shuffle:
      ds = ds.shuffle(rng.fold_in("shuffle").as_seed())
    return ds
