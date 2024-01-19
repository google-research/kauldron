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

"""Data source."""

import dataclasses
import functools
import sys

import jax
from kauldron import kd
from projects.nerf.core import structs
from projects.nerf.data import base
import numpy as np


@dataclasses.dataclass(frozen=True, kw_only=True)
class RaySampler(base.DataSource):
  """Sample random rays across the full images.

  Attributes:
    num_samples: Global number of rays to samples (across all hosts). Each host
      sample `num_samples // num_hosts` ray.
  """

  num_samples: int

  # TODO(epot): Refactor into a `grain.BatchRandomDataSource`, that uses
  # `__getitem__(self, record_keys: i32['batch_size'])`, rather than
  # `__getitem__(self, record_keys: int)`

  @functools.cached_property
  def batch_size(self) -> kd.data.BatchSize:
    return kd.data.BatchSize(self.num_samples)

  @functools.cached_property
  def seed_for_process(self):
    # Each host has a separate rng generator !!!
    # Note that in practice, `record_key` should already be different for
    # each hosts.
    # Each worker get the same seed, but get a different `record_key`, so should
    # be good.
    ss = np.random.SeedSequence(self.seed)
    seed = ss.spawn(jax.process_count())[jax.process_index()]
    # TODO(epot): `key` should be `seed`, not `seed.entropy`
    return np.random.Philox(key=seed.entropy)

  def __getitem__(self, record_key: int) -> structs.Batch:
    # TODO(epot): because each host call `rng.choice` independently, one
    # batch might contain the same ray multiple times.
    rng = np.random.Generator(self.seed_for_process.jumped(record_key))
    indices = rng.choice(
        len(self.flat_batch),
        size=self.batch_size.per_process,
        replace=False,
        shuffle=True,
    )
    return self.flat_batch[indices]

  def __len__(self) -> int:
    return sys.maxsize


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImageSampler(base.DataSource):
  """Sample full images."""

  def __getitem__(self, record_key: int) -> structs.Batch:
    # Returns the full image
    return self.batch[record_key]

  def __len__(self) -> int:
    return len(self.batch)
