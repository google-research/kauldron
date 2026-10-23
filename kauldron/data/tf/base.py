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
from jax.experimental import multihost_utils
from kauldron import random
from kauldron.data import iterators
from kauldron.data import pipelines
from kauldron.data.tf import grain_utils
from kauldron.data.tf import transform_utils as tr_utils
from kauldron.ktyping import PRNGKey, PRNGKeyLike, PyTree, TfArray  # pylint: disable=g-importing-member,g-multiple-import
import tensorflow as tf
import tensorflow_datasets as tfds

# pylint: disable=logging-fstring-interpolation

PADDING_MASK_KEY: str = "__is_padding__"
DropRemainder = pipelines.DropRemainder


# Output of `tfds.as_numpy`
_NpArray: TypeAlias = Any

# Sentinel value that indicate the field should only be activated at the top
# level node.
_ROOT_ONLY: Any = object()

_Transforms = Sequence[grain.Transformation] | dict[str, grain.Transformation]


@dataclasses.dataclass(frozen=True, kw_only=True, eq=True)
class TFDataPipeline(pipelines.Pipeline, abc.ABC):
  """Base classes for all `tf.data` pipeline.

  Do NOT instantiate this class directly. Instead, use one of the subclasses
  below or create your own.

  Subclasses should only implement the `ds_for_current_process` method which
  returns the `tf.data.Dataset` for the current process.

  Attributes:
    batch_drop_remainder: Determines how to handle a final partial batch of a
      dataset (DROP, KEEP, PAD, PAD_WITH_MASK).
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
    batch_size: int | None = ...  # pyrefly: ignore[bad-assignment]
    seed: PRNGKeyLike | None = ...  # pyrefly: ignore[not-a-type]

  batch_drop_remainder: bool | str | DropRemainder = True
  transforms: _Transforms = dataclasses.field(default_factory=tuple)

  # Those fields are only applied once at the top level
  tf_data_options: Optional[tf.data.Options] = None
  prefetch_size: Optional[int] = _ROOT_ONLY
  checkpoint: bool | None = None

  # ======================== Protocol ========================

  def __post_init__(self):
    object.__setattr__(
        self,
        "batch_drop_remainder",
        pipelines.normalize_batch_drop_remainder(self.batch_drop_remainder),
    )

    if hasattr(super(), "__post_init__"):
      super().__post_init__()  # Future proof to run `__post_init__` in parents  # pyrefly: ignore[missing-attribute]

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
  def ds_for_current_process(self, rng: PRNGKey) -> tf.data.Dataset:
    """Returns the dataset for the current process.

    Args:
      rng: A PRNGKey used to sample the dataset. The `rng` is the same for all
        processes, so it's the users responsibility to call
        `jax.random.fold_in(rng, jax.process_index())` if required.

    Returns:
      The `tf.data.Dataset` for the current process. Each process should yield
        non-overlapping examples.
    """
    raise NotImplementedError

  def transform_ds(
      self, ds: tf.data.Dataset, *, rng: PRNGKey
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
  def element_spec(self) -> PyTree[enp.ArraySpec]:  # pyrefly: ignore[not-a-type]
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
      rng = jax.random.PRNGKey(self.seed)
      rng = random.fold_in_str(rng, "kmix")
      ds = self.ds_with_transforms(rng, _is_root=True)

    # Drop grain meta features
    ds = ds.map(lambda ex: grain_utils.split_grain_meta_features(ex)[1])
    return ds

  def ds_with_transforms(
      self, rng: PRNGKey, *, _is_root: bool = False
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
      match self.batch_drop_remainder:
        case DropRemainder.DROP:
          transforms.append(
              grain.TfBatch(
                  batch_size=self.host_batch_size,
                  drop_remainder=True,
              )
          )
        case DropRemainder.KEEP:
          transforms.append(
              grain.TfBatch(
                  batch_size=self.host_batch_size,
                  drop_remainder=False,
              )
          )
        case DropRemainder.PAD:
          transforms.append(
              _TfBatchWithPad(
                  batch_size=self.host_batch_size,
                  add_padding_mask=False,
              )
          )
        case DropRemainder.PAD_WITH_MASK:
          transforms.append(
              _TfBatchWithPad(
                  batch_size=self.host_batch_size,
                  add_padding_mask=True,
              )
          )
        case _:
          raise ValueError(
              f"Unsupported batch_drop_remainder: {self.batch_drop_remainder}"
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TfBatchWithPad(grain.UnsafeTfDataTransform):
  """Batches a `tf.data.Dataset` and pads partial batches with zeros.

  Attributes:
    batch_size: Target per-host batch size.
    add_padding_mask: Whether to inject a boolean `__is_padding__` mask (`False`
      for valid elements, `True` for padded elements) and equalize batch counts
      across hosts when cardinality is known.
    mask_key: Feature key used for the boolean padding mask when
      `add_padding_mask=True`.
    num_parallel_calls: Number of parallel calls for batching and mapping.
  """

  batch_size: int
  add_padding_mask: bool = False
  mask_key: str = PADDING_MASK_KEY
  num_parallel_calls: int = tf.data.AUTOTUNE

  def apply_to_dataset(self, ds: tf.data.Dataset) -> tf.data.Dataset:
    """Applies batching, zero-padding, and optional mask injection to `ds`."""
    # Validate any pre-existing padding mask feature before batching.
    if self.add_padding_mask and self.mask_key in ds.element_spec:
      spec = ds.element_spec[self.mask_key]
      if spec.dtype != tf.bool or spec.shape.rank != 0:
        raise ValueError(
            f"Dataset already contains {self.mask_key!r} with spec {spec}, "
            "expected scalar tf.bool."
        )

    cardinality = ds.cardinality().numpy()
    if self.add_padding_mask and cardinality >= 0:
      # When dataset cardinality is statically known and
      # `add_padding_mask=True`, synchronize local shard lengths across JAX
      # processes and pad the unbatched dataset up to a common multiple of
      # `batch_size`. This guarantees that every host produces the exact same
      # number of batches even when sharding yields uneven per-host dataset
      # lengths.
      local_cardinality = int(cardinality)
      if jax.process_count() > 1:
        max_cardinality = int(
            multihost_utils.process_allgather(local_cardinality).max()
        )
      else:
        max_cardinality = local_cardinality
      # Round `max_cardinality` up to the next multiple of `self.batch_size`.
      target_cardinality = (
          -(-max_cardinality // self.batch_size) * self.batch_size
      )
      num_pad_elements = target_cardinality - local_cardinality

      # Tag all valid dataset elements with `mask_key = False` if not already
      # present.
      if self.mask_key not in ds.element_spec:
        ds = ds.map(
            lambda features: dict(features)
            | {self.mask_key: tf.constant(False, dtype=tf.bool)},
            num_parallel_calls=self.num_parallel_calls,
        )

      # Append `num_pad_elements` zero-filled filler elements with
      # `mask_key = True` and `grain.INDEX = -1` so downstream metrics can
      # discard padded slots and Grain's validator sees a valid index feature.
      if num_pad_elements > 0:
        filler = jax.tree.map(
            lambda spec: tf.zeros(spec.shape, dtype=spec.dtype),
            ds.element_spec,
        )
        filler[self.mask_key] = tf.constant(True, dtype=tf.bool)
        if grain.INDEX in filler:
          filler[grain.INDEX] = tf.cast(-1, tf.int64)
        pad_ds = tf.data.Dataset.from_tensors(filler).repeat(num_pad_elements)
        ds = ds.concatenate(pad_ds)

      # Since the dataset length is now an exact multiple of `batch_size`,
      # `drop_remainder=True` preserves all elements while setting a static
      # leading dimension of `batch_size` on `element_spec`.
      return ds.batch(
          self.batch_size,
          drop_remainder=True,
          num_parallel_calls=self.num_parallel_calls,
      )

    # Fallback path when `add_padding_mask=False` (`DropRemainder.PAD`) or when
    # dataset cardinality is unknown/infinite (e.g. after `.filter()`): batch
    # with `drop_remainder=False` and dynamically pad any partial batch to
    # `self.batch_size` inside `ds.map`.
    ds = ds.batch(
        self.batch_size,
        drop_remainder=False,
        num_parallel_calls=self.num_parallel_calls,
    )

    def _pad_batch(
        batch: dict[str, TfArray["b *_"]],  # pyrefly: ignore[not-a-type, unknown-name]
    ) -> dict[str, TfArray["B *_"]]:  # pyrefly: ignore[not-a-type, unknown-name]
      # Infer the runtime batch size from the first tensor leaf that has a
      # defined rank >= 1 (skipping `None`, scalars, or unranked tensors).
      first_leaf = _first_shaped_tensor(batch)
      valid_batch_size = tf.shape(first_leaf)[0]
      pad_size = self.batch_size - valid_batch_size

      def _pad_leaf(
          leaf: TfArray["b *_"],  # pyrefly: ignore[not-a-type, unknown-name]
      ) -> TfArray["B *_"]:  # pyrefly: ignore[not-a-type, unknown-name]
        # Pad the batch dimension (axis 0) with zeros up to `self.batch_size`
        # and restore the static leading dimension when the leaf rank is known.
        pad_shape = tf.concat([[pad_size], tf.shape(leaf)[1:]], axis=0)
        padded = tf.concat(
            [leaf, tf.zeros(pad_shape, dtype=leaf.dtype)], axis=0
        )
        if leaf.shape.rank is not None:
          padded.set_shape((self.batch_size,) + tuple(leaf.shape[1:]))
        return padded

      padded_batch = jax.tree.map(_pad_leaf, batch)
      if self.add_padding_mask:
        # Mark indices `>= valid_batch_size` as padded (`True`), OR-ing with any
        # pre-existing `mask_key` tensor in the batch.
        is_padded = tf.range(self.batch_size) >= valid_batch_size
        if self.mask_key in batch:
          padded_batch[self.mask_key] = tf.logical_or(
              padded_batch[self.mask_key], is_padded
          )
        else:
          padded_batch[self.mask_key] = is_padded
        padded_batch[self.mask_key].set_shape((self.batch_size,))
        # Replace padded `grain.INDEX` entries with `-1` to match the
        # known-cardinality padding convention.
        if grain.INDEX in padded_batch:
          padded_batch[grain.INDEX] = tf.where(
              is_padded,
              tf.cast(-1, tf.int64),
              padded_batch[grain.INDEX],
          )
          padded_batch[grain.INDEX].set_shape((self.batch_size,))

      return padded_batch

    return ds.map(_pad_batch, num_parallel_calls=self.num_parallel_calls)


def _first_shaped_tensor(
    batch: PyTree[TfArray["*b"]],  # pyrefly: ignore[not-a-type]
) -> TfArray["b *_"]:  # pyrefly: ignore[not-a-type, unknown-name]
  """Returns the first tensor leaf in `batch` with a defined rank >= 1.

  Args:
    batch: Batch PyTree of tensors.

  Returns:
    The first `TfArray` leaf with `leaf.shape.rank is not None` and
    `leaf.shape.rank >= 1`.

  Raises:
    ValueError: If no tensor leaf in `batch` has a defined rank >= 1.
  """
  for leaf in jax.tree.leaves(batch):
    if (
        isinstance(leaf, TfArray)
        and leaf.shape.rank is not None
        and isinstance(leaf, TfArray["_ *_"])
    ):
      return leaf
  raise ValueError(
      f"Could not find any tensor leaf with rank >= 1 in batch: {batch}"
  )
