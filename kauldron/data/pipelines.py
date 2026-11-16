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

"""Base data pipeline."""

import abc
import dataclasses
import enum
import functools
import random
from typing import Any, Optional, TypeAlias

from etils import edc
from etils import enp
from etils.etree import jax as etree  # pylint: disable=g-importing-member
from kauldron.data import data_utils
from kauldron.data import iterators
from kauldron.data import utils
from kauldron.ktyping import PRNGKeyLike, PyTree  # pylint: disable=g-importing-member,g-multiple-import
from kauldron.utils import config_util

# Output of `tfds.as_numpy`
_NpTfdsDataset: TypeAlias = Any
_NpArray: TypeAlias = Any


class DropRemainder(enum.StrEnum):
  """Determines how to handle a final partial batch of a dataset.

  Attributes:
    DROP: Drop the last partial batch.
    KEEP: Keep the last partial batch as is (may have size < `batch_size`).
    PAD: Pad the last partial batch with zeros to reach `batch_size`.
      WARNING: Users are responsible for handling the extra zero-padded elements
        in downstream models/metrics (e.g., via masking or sentinel filtering).
        Additionally, in multi-host/multi-process pipelines, dataset length is
        not equalized before sharding, which can cause different hosts to yield
        an uneven number of batches and lead to distributed step
        desynchronization.
    PAD_WITH_MASK: Like PAD, but equalizes batch counts across hosts when
      dataset cardinality is known and tags batches with an `__is_padding__`
      mask (or wraps elements in `PaddedElement` in
      `kd.data.py.DataSourceBase`).
      WARNING: In `kd.data.py`, downstream readers and transforms must
        explicitly inspect the `is_padding` field on `PaddedElement` and handle
        `None` elements (where `is_padding=True`) correctly (e.g. substituting
        dummy bytes for parsing and tagging batches with an `__is_padding__`
        mask so metrics can discard them). In `kd.data.tf`, batches are
        automatically zero-padded and tagged with `"__is_padding__"`.
  """

  DROP = enum.auto()
  KEEP = enum.auto()
  PAD = enum.auto()
  PAD_WITH_MASK = enum.auto()


def normalize_batch_drop_remainder(
    batch_drop_remainder: bool | str | DropRemainder,
) -> DropRemainder:
  """Normalize the batch drop remainder to an enum value."""
  if batch_drop_remainder is True:  # pylint: disable=g-bool-id-comparison
    return DropRemainder.DROP
  elif batch_drop_remainder is False:  # pylint: disable=g-bool-id-comparison
    return DropRemainder.KEEP
  else:
    return DropRemainder(batch_drop_remainder)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class Pipeline(data_utils.IterableDataset, config_util.UpdateFromRootCfg):
  """Base class for kauldron data pipelines.

  Subclasses should implement:

  * `__iter__`: Yield individual batches
  * (optionally) `__len__`: Number of iterations

  Subclasses are responsible for:

  * batching
  * shuffling
  * sharding: Each host yields different examples

  Attributes:
    batch_size: Global batch size. Has to be divisible by number of global
      devices. Pipeline should take care of sharding the data between hosts.
      Setting to `0` disables batching.
    seed: Random seed to be used for things like shuffling and randomness in
      preprocessing. Defaults to the seed from the root config.
  """

  batch_size: int | None = None
  seed: Optional[PRNGKeyLike] = config_util.ROOT_CFG_REF.seed  # pyrefly: ignore[not-a-type]

  @functools.cached_property
  def element_spec(self) -> PyTree[enp.ArraySpec]:  # pyrefly: ignore[not-a-type]
    """Returns the element specs of a single batch."""
    first_elem = next(iter(self))
    return etree.spec_like(first_elem)

  @functools.cached_property
  def host_batch_size(self) -> int:
    return utils.BatchSize(self.batch_size).per_process  # pyrefly: ignore[bad-argument-type]

  # Overwrite the base class as the signature change.
  @abc.abstractmethod
  def __iter__(self) -> iterators.Iterator:
    """Iterate over the dataset elements."""
    raise NotImplementedError()

  __repr__ = edc.repr

  # For convenience, it can be annoying to have to manually set the seed when
  # debugging on Colab.
  def _assert_root_cfg_resolved(self) -> None:
    # If the seed is not set when the pipeline is created, we create an
    # arbitrary random seed.
    if isinstance(self.seed, config_util._FakeRootCfg):  # pylint: disable=protected-access
      object.__setattr__(self, 'seed', random.randint(0, 1000000000))
    super()._assert_root_cfg_resolved()
