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

"""`tf.data` data pipeline."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import typing
from typing import Any, Optional, TypeAlias

from absl import logging
from etils import enp
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import grain.tensorflow as grain
import jax
from kauldron import random
from kauldron.data import iterators
from kauldron.data import pipelines
from kauldron.data.tf import grain_utils
from kauldron.data.tf import transform_utils as tr_utils
from kauldron.typing import PRNGKeyLike, PyTree  # pylint: disable=g-importing-member,g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds

# pylint: disable=logging-fstring-interpolation


# Output of `tfds.as_numpy`
_NpArray: TypeAlias = Any

# Sentinel value that indicate the field should only be activated at the top
# level node.
_ROOT_ONLY: Any = object()

_Transforms = Sequence[grain.Transformation] | dict[str, grain.Transformation]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class TFDataPipeline(pipelines.Pipeline, abc.ABC):
  """Base classes for all `tf.data` pipeline.

  Do NOT use this class directly. Instead, use one of the subclasses.

  Subclasses should only implement the `ds_for_current_process` method which
  returns the `tf.data.Dataset` for the current process.

  Attributes:
    transforms: A list of `grain.Transformation` to apply to the dataset. Can be
      a dict to allow easier CLI / sweep access (
      `--cfg.train_ds.transforms.img_scale.in_vrange=(-1,1)`)
    tf_data_options: An optional tf.data.Options instance to be applied to the
      dataset.
    prefetch_size: Number of batches to prefetch for this dataset. Defaults to
      AUTOTUNE.
    checkpoint: Whether the pipeline should be checkpointed. By default,
      pipelines are checkpointed if they supports symbolic checkpointing, and
      not otherwise.
  """

  # TODO(epot): Need to duplicate `batch_size` because pytype fail to detect
  # the parent class
  if typing.TYPE_CHECKING:
    batch_size: int | None = ...
    seed: PRNGKeyLike | None = ...

  # TODO(epot): Users should also be able to specify drop_reminder or mask
  batch_drop_remainder: bool = True
  transforms: _Transforms = dataclasses.field(default_factory=tuple)

  # Those fields are only applied once at the top level
  tf_data_options: Optional[tf.data.Options] = None
  prefetch_size: Optional[int] = _ROOT_ONLY
  checkpoint: bool | None = None

  # ======================== Protocol ========================

  def __post_init__(self):
    if hasattr(super(), "__post_init__"):
      super().__post_init__()  # Future proof to run `__post_init__` in parents

    if not self._supports_symbolic_checkpoint:
      if self.checkpoint is None:
        logging.info(
            f"{type(self).__qualname__} does not support symbolic"
            " checkpointing, so dataset won't be checkpointed, unless"
            " `checkpoint=` is explicitly set."
        )
      elif self.checkpoint:
        logging.info(
            f"{type(self).__qualname__} does not support symbolic"
            " checkpointing, but `checkpoint=True`. This might lead to huge"
            " checkpoints."
        )
      # Else, `checkpoint` is explicitly set to False, so nothing to do.

  @abc.abstractmethod
  def ds_for_current_process(self, rng: random.PRNGKey) -> tf.data.Dataset:
    """Returns the dataset for the current process.

    Args:
      rng: A PRNGKey used to sample the dataset. The `rng` is the same for all
        processes, so it's the users responsibility to call
        `rng.fold_in(jax.process_index())` if required.

    Returns:
      The `tf.data.Dataset` for the current process. Each process should yield
        non-overlapping examples.
    """
    raise NotImplementedError

  def transform_ds(
      self, ds: tf.data.Dataset, *, rng: random.PRNGKey
  ) -> tf.data.Dataset:
    """Eventually apply transforms to the dataset."""
    ds = grain_utils.maybe_add_grain_meta_features(
        ds,
        rng=rng,
    )

    ds = self._apply_transforms(ds)
    return ds

  # ======================== Public API ========================

  def __iter__(self) -> iterators.Iterator:
    """Iterate over the dataset elements."""
    # Some custom ops raise an error when checkpointed. We could try to
    # auto-detect, however it might be better to explicitly raise an error
    # so user explicitly set their pipeline as non-checkpointable.
    if self._should_checkpoint:
      return iterators.TFDataIterator(source=self, iter=iter(self._root_ds))
    else:
      return iterators.NonCheckpointableIterator(
          source=self, iter=iter(tfds.as_numpy(self._root_ds))
      )

  @property
  def element_spec(self) -> PyTree[enp.ArraySpec]:
    """Returns the element specs of the dataset."""
    return etree.spec_like(self._root_ds.element_spec)

  def __len__(self) -> int:
    return len(self._root_ds)

  # ======================== Internals ========================

  @functools.cached_property
  def _root_ds(self) -> tf.data.Dataset:
    """Returns a numpy tf.data.Dataset iterator."""
    self._assert_root_cfg_resolved()

    # Loader sometimes uses `jax.random` to generate integer seeds, so allow
    # host<>device here.
    # Instead could force the `PRNGKey` to stay in the CPU.
    with jax.transfer_guard("allow"):
      rng = random.PRNGKey(self.seed)
      rng = rng.fold_in("kmix")
      ds = self.ds_with_transforms(rng, _is_root=True)

    # Drop grain meta features
    ds = ds.map(lambda ex: grain_utils.split_grain_meta_features(ex)[1])
    return ds

  def ds_with_transforms(
      self, rng: random.PRNGKey, *, _is_root: bool = False
  ) -> tf.data.Dataset:
    """Create the `tf.data.Dataset` and apply all the transforms."""
    ds = self.ds_for_current_process(rng)
    ds = self.transform_ds(ds, rng=rng)

    # Additional transformations applied only at the top level
    ds = self._apply_options(ds, is_root=_is_root)

    # Only apply `prefetch` at the end of the pipeline.
    prefetch_size = _get_if_root(
        self.prefetch_size,
        default=tf.data.AUTOTUNE,
        is_root=_is_root,
    )
    if prefetch_size:
      ds = ds.prefetch(prefetch_size)

    return ds

  def _apply_transforms(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    """Apply transforms to the dataset."""

    transforms = []
    if isinstance(self.transforms, Mapping):
      transforms.extend(self.transforms.values())
    else:
      transforms.extend(self.transforms)
    if self.batch_size:
      transforms.append(
          grain.TfBatch(
              batch_size=self.host_batch_size,
              drop_remainder=self.batch_drop_remainder,
          )
      )
    ds = tr_utils.apply_transformations(ds, transforms)
    return ds

  def _apply_options(
      self, ds: tf.data.Dataset, *, is_root: bool = False
  ) -> tf.data.Dataset:
    """Apply `tf.data.Options` to the dataset."""
    if is_root:
      # Default data options (can be overwritten using tf_data_options) obtained
      # from https://github.com/google/CommonLoopUtils/tree/HEAD/clu/deterministic_data.py
      ds_options = tf.data.Options()
      ds_options.experimental_optimization.map_parallelization = True
      ds_options.threading.private_threadpool_size = 48
      ds_options.threading.max_intra_op_parallelism = 1
      # Start fetching the data as soon as the `tf.data` pipeline is created
      # (instead of in the first `next(iter(ds))` call) to speed up start time.
      ds_options.experimental_warm_start = True

      if self._supports_symbolic_checkpoint:
        ds_options.experimental_symbolic_checkpoint = True
    else:
      ds_options = None

    if self.tf_data_options is not None:
      if is_root:
        assert ds_options is not None
        ds_options = ds_options.merge(self.tf_data_options)
      else:
        ds_options = self.tf_data_options

    if ds_options is not None:
      ds = ds.with_options(ds_options)
    return ds

  # Whether the pipeline supports symbolic checkpointing.
  # This should be defined in the subclasses.
  @property
  @abc.abstractmethod
  def _supports_symbolic_checkpoint(self) -> bool:
    raise NotImplementedError(
        "Abstract property `_supports_symbolic_checkpoint` not implemented for"
        f" {type(self)}"
    )

  @property
  def _should_checkpoint(self) -> bool:
    """Whether the pipeline should be checkpointed."""
    if self.checkpoint is None:
      return self._supports_symbolic_checkpoint
    else:
      return self.checkpoint


def _get_if_root(value, *, default, is_root: bool):
  if value is _ROOT_ONLY:
    if is_root:
      return default
    else:
      return None
  else:
    return value
