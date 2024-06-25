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

import dataclasses
import functools

from kauldron import random
from kauldron.data.kmix import base
import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class SampleFromDatasets(base.TFDataPipeline):
  """Dataset mixture.

  Attributes are forwarded to `tf.data.Dataset.sample_from_datasets`.

  Attributes:
    datasets: List of datasets to sample from.
    weights: Weights for each dataset. If None, all datasets are sampled with
      equal weights. Weights are normalized to sum to 1. So sampling among [2,
      4, 2] is the same as [0.25, 0.5, 0.25].
    stop_on_empty_dataset: If True, the iteration will stop on the first empty
      dataset.
    rerandomize_each_iteration: If True, the mixture will reshuffle the datasets
      each time it is iterated over.
  """

  datasets: list[base.TFDataPipeline]
  _: dataclasses.KW_ONLY
  weights: None | list[float] = None
  stop_on_empty_dataset: bool = False
  rerandomize_each_iteration: bool = True

  def __post_init__(self):
    if not all(isinstance(ds, base.TFDataPipeline) for ds in self.datasets):
      raise ValueError(
          'All datasets in `SampleFromDatasets` should inherit from'
          ' `TFDataPipeline`.'
      )

  def ds_for_current_process(self, rng: random.PRNGKey) -> tf.data.Dataset:
    # We make sure each nested dataset get a different seed
    datasets = [
        ds.ds_with_transforms(rng.fold_in(i))
        for i, ds in enumerate(self.datasets)
    ]

    ds = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=self.weights,
        seed=int(rng.fold_in('sample_from_datasets').bits()),
        stop_on_empty_dataset=self.stop_on_empty_dataset,
        rerandomize_each_iteration=self.rerandomize_each_iteration,
    )
    return ds

  @functools.cached_property
  def _supports_symbolic_checkpoint(self) -> bool:
    return all(ds._supports_symbolic_checkpoint for ds in self.datasets)  # pylint: disable=protected-access
