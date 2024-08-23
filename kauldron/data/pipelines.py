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

"""Data pipelines."""

import abc
import dataclasses
import functools
from typing import Any, Optional, TypeAlias

from absl import logging
from etils import edc
from etils import enp
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import grain.python as pygrain
from kauldron.data import data_utils
from kauldron.data import iterators
from kauldron.data import utils
from kauldron.typing import PRNGKeyLike, PyTree  # pylint: disable=g-importing-member,g-multiple-import
from kauldron.utils import config_util

# Output of `tfds.as_numpy`
_NpTfdsDataset: TypeAlias = Any
_NpArray: TypeAlias = Any


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class Pipeline(data_utils.IterableDataset, config_util.UpdateFromRootCfg):
  """Base class for kauldron data pipelines.

  Subclasses should implement:

  * `__iter__`: Yield individual batches
  * (optionally) `__len__`: Number of iterations

  Subclasses are responsible for:

  * batching
  * shuflling
  * sharding: Each host yield different examples

  Attributes:
    batch_size: Global batch size. Has to be divisible by number of global
      devices. Pipeline should take care of sharding the data between hosts.
      Setting to `0` disable batching.
    seed: Random seed to be used for things like shuffling and randomness in
      preprocessing. Defaults to the seed from the root config.
  """

  batch_size: int | None
  seed: Optional[PRNGKeyLike] = config_util.ROOT_CFG_REF.seed

  @functools.cached_property
  def element_spec(self) -> PyTree[enp.ArraySpec]:
    """Returns the element specs of a single batch."""
    first_elem = next(iter(self))
    return etree.spec_like(first_elem)

  @functools.cached_property
  def host_batch_size(self) -> int:
    return utils.BatchSize(self.batch_size).per_process

  # Overwrite the base class as the signature change.
  @abc.abstractmethod
  def __iter__(self) -> iterators.Iterator:
    """Iterate over the dataset elements."""
    raise NotImplementedError()

  __repr__ = edc.repr


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class PyGrainPipeline(Pipeline):
  """Basic pygrain pipeline.

  Attributes:
    data_source: a random access datasource
    transforms: A list of transformations to apply to the dataset. Each
      transformation should be either a `grain.MapTransform` or a
      `grain.RandomMapTransform`.
    shuffle: whether to shuffle
    num_epochs: Number of epoch. If missing, number of iteration is given by
      `cfg.num_training_steps`
    worker_count: how many worker processes to use for data loading
  """

  data_source: pygrain.RandomAccessDataSource
  shuffle: bool
  # TODO(epot): More consistent way to customize the number of steps. Unify:
  # * `cfg.num_training_steps`
  # * `cfg.train_ds.num_epochs` (for `PyGrainPipeline`)
  # * `cfg.train_ds.loader.num_epochs` (for `kd.data.loader.GrainTfds`)
  # * `cfg.evals[].num_batches`
  num_epochs: Optional[int] = None
  transforms: pygrain.Transformations = dataclasses.field(default_factory=list)
  worker_count: int = 16
  num_prefetch_elements: int = 1

  @functools.cached_property
  def batch_fn(self) -> pygrain.Batch:
    """Batch transformaton."""
    # TODO(klausg): Users should also be able to specify drop_reminder or mask
    return pygrain.Batch(self.host_batch_size, drop_remainder=True)

  @property
  def sampler(self) -> pygrain.Sampler:
    return pygrain.IndexSampler(
        num_records=len(self.data_source),
        num_epochs=self.num_epochs,
        seed=self.seed,
        shuffle=self.shuffle,
        shard_options=pygrain.ShardByJaxProcess(),
    )

  @functools.cached_property
  def loader(self) -> pygrain.DataLoader:
    """Returns a numpy tf.data.Dataset iterator."""
    self._assert_root_cfg_resolved()

    transformations = []
    transformations.extend(self.transforms)
    if self.batch_size:
      transformations.append(self.batch_fn)

    worker_count = self.worker_count
    if epy.is_notebook():  # in colab worker_count has to be 0
      # TODO(klausg): autodetect if Kernel supports multiprocessing
      # Could check
      # from multiprocessing import spawn
      # spawn.get_executable() is not None
      print("Disabling pygrain multi-processing (unsupported in colab).")
      worker_count = 0

    logging.info(
        "Creating pygrain.Dataloader from %s with %d elements",
        self.data_source,
        len(self.data_source),
    )
    dataloader = pygrain.DataLoader(
        data_source=self.data_source,
        operations=transformations,
        sampler=self.sampler,
        worker_count=worker_count,
        worker_buffer_size=self.num_prefetch_elements,
        shard_options=pygrain.ShardByJaxProcess(),
    )
    return dataloader

  def __iter__(self) -> iterators.Iterator:
    """Iterate over the dataset elements."""
    return iterators.PyGrainIterator(source=self, iter=iter(self.loader))

  def __len__(self) -> int:
    if self.num_epochs is None:
      raise TypeError("Cannot get length of infinite dataset.")
    else:
      return self.num_epochs * len(self.data_source)
