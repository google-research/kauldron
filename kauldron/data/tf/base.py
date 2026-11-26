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
from collections.abc import Mapping
import dataclasses
import functools
import typing
from typing import Any, Optional, TypeAlias

from absl import logging
from clu import preprocess_spec
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
from kauldron.data.transforms import base as tr_base
from kauldron.ktyping import PRNGKey, PRNGKeyLike, PyTree, TfArray  # pylint: disable=g-importing-member,g-multiple-import
import numpy as np
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

_Transforms: TypeAlias = tr_utils.Transformations


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
    transforms: A list of `grain.Transformation` to apply to the dataset before
      batching. Can be a dict to allow easier CLI / sweep access (
      `--cfg.train_ds.transforms.img_scale.in_vrange=(-1,1)`)
    post_batch_transforms: A list of `grain.Transformation` to apply to the
      dataset after batching. Can be a dict to allow easier CLI / sweep access (
      `--cfg.train_ds.post_batch_transforms.img_scale.in_vrange=(-1,1)`).
      Requires active batching via `batch_size`, a batch transform in
      `transforms`, or a pre-batched dataset source (`is_prebatched=True`). Each
      post-batch `grain.RandomMapTransform` receives one seed per host batch, so
      transforms such as `kd.data.tf.RandomCrop` share one random draw across
      the batch. `kd.data.tf.OneHot` defaults to `multi=True` and collapses a
      `(B,)` label vector into one multi-hot vector unless you pass
      `multi=False`.
    is_prebatched: Whether `ds_for_current_process` already yields batched
      elements (for example when wrapping a pre-batched non-Grain
      `tf.data.Dataset` with `batch_size=None`). When `True`, all non-`None`
      element leaves must have `rank >= 1` and matching non-zero leading batch
      dimensions (`TensorSpec`, `RaggedTensorSpec`, and `SparseTensorSpec` are
      supported), and Grain metadata features such as `_index` are attached with
      a 1D `(B,)` batch dimension. Defaults to `None` (auto-detected by built-in
      pre-batched pipelines such as `ElementSpecDataset` and `ZipDatasets`, and
      `False` otherwise).
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
  post_batch_transforms: _Transforms = dataclasses.field(default_factory=tuple)
  is_prebatched: bool | None = None

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
    if self.is_prebatched is None:
      for cls in type(self).__mro__:
        if cls is TFDataPipeline:
          break
        cls_val = cls.__dict__.get("is_prebatched")
        if isinstance(cls_val, bool):
          object.__setattr__(self, "is_prebatched", cls_val)
          break

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
        is_prebatched=self._should_treat_as_prebatched(ds),
    )

    ds = self._apply_transforms(ds, rng=rng)
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

  @property
  def _drop_remainder_mode(self) -> DropRemainder:
    """Returns `batch_drop_remainder` normalized to a `DropRemainder` enum."""
    return pipelines.normalize_batch_drop_remainder(self.batch_drop_remainder)

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

  def _is_prebatched_source(self, ds: tf.data.Dataset) -> bool:
    """Returns whether this pipeline's source dataset is already batched."""
    del ds
    return False

  def _should_treat_as_prebatched(self, ds: tf.data.Dataset) -> bool:
    """Returns whether `ds` from `ds_for_current_process` is pre-batched."""
    if self.is_prebatched is not None:
      return self.is_prebatched
    return self._is_prebatched_source(ds)

  def _is_batched_ds(self, ds: tf.data.Dataset) -> bool:
    """Returns whether `ds` represents a batched dataset."""
    return _has_batched_index(ds)

  def _apply_transforms(
      self,
      ds: tf.data.Dataset,
      *,
      rng: PRNGKeyLike | None = None,  # pyrefly: ignore[not-a-type]
  ) -> tf.data.Dataset:
    """Apply transforms to the dataset."""
    transforms = _normalize_transforms(self.transforms)
    post_batch_transforms = _normalize_transforms(self.post_batch_transforms)

    if self.batch_size:
      mode = self._drop_remainder_mode
      if mode in (DropRemainder.DROP, DropRemainder.KEEP):
        transforms.append(
            grain.TfBatch(
                batch_size=self.host_batch_size,
                drop_remainder=(mode == DropRemainder.DROP),
            )
        )
      else:
        transforms.append(
            _TfBatchWithPad(
                batch_size=self.host_batch_size,
                add_padding_mask=(mode == DropRemainder.PAD_WITH_MASK),
            )
        )
    ds = tr_utils.apply_transformations(ds, transforms)

    if post_batch_transforms:
      if not self._is_batched_ds(ds):
        raise ValueError(
            "Cannot apply `post_batch_transforms` when batching is disabled"
            " (`batch_size` is None or 0, the dataset source is not"
            " pre-batched, and `transforms` does not produce a batched"
            " dataset)."
        )

      post_batch_key: random.PRNGKey | None = None

      def _get_base_seed(step_idx: int) -> tuple[int, int]:
        nonlocal post_batch_key
        with jax.transfer_guard("allow"):
          if post_batch_key is None:
            if rng is not None:
              seed_source = rng
            elif isinstance(
                self.seed, (int, np.integer, jax.Array, random.PRNGKey)
            ):
              seed_source = self.seed
            else:
              seed_source = 0
            post_batch_key = (
                random.PRNGKey(seed_source)
                .fold_in("post_batch_transforms")
                .fold_in(jax.process_index())
            )
          subkey = post_batch_key.fold_in(step_idx)
          raw_bits = np.asarray(
              jax.random.bits(subkey.rng, shape=(2,), dtype=jax.numpy.uint32)
          )
          return (
              int(raw_bits.view(np.int32)[0]),
              int(raw_bits.view(np.int32)[1]),
          )

      any_allow_drop = False
      adapted_post_batch: list[grain.Transformation] = []
      for idx, raw_tr in enumerate(post_batch_transforms):
        adapted = tr_utils.adapt_for_tfgrain(raw_tr)
        if isinstance(adapted, grain.RandomMapTransform):
          adapted = _PostBatchRandomMapTransform(
              transform=adapted,
              base_seed=_get_base_seed(idx),
          )
        elif isinstance(
            raw_tr,
            (preprocess_spec.PreprocessFn, preprocess_spec.RandomMapTransform),
        ):
          adapted = _PostBatchPreprocessFnTransform(
              transform=raw_tr,
              base_seed=_get_base_seed(idx),
          )
        if isinstance(adapted, grain.MapTransform):
          allow_drop = isinstance(raw_tr, tr_base.Elements) and (
              PADDING_MASK_KEY in raw_tr.drop
              or PADDING_MASK_KEY in raw_tr.rename
          )
          any_allow_drop = any_allow_drop or allow_drop
          adapted = _PreservePaddingMaskMapTransform(
              transform=adapted,
              allow_drop=allow_drop,
          )
        adapted_post_batch.append(adapted)

      ds = tr_utils.apply_transformations(ds, adapted_post_batch)
      _validate_padding_mask_spec(ds.element_spec)
      if (
          self._drop_remainder_mode == DropRemainder.PAD_WITH_MASK
          and jax.process_count() > 1
      ):
        post_cardinality = int(ds.cardinality().numpy())
        if post_cardinality < 0:
          raise ValueError(
              "DropRemainder.PAD_WITH_MASK across multiple JAX processes"
              " requires known non-negative dataset cardinality after"
              " `post_batch_transforms`, but got"
              f" cardinality={post_cardinality} and"
              f" element_spec={ds.element_spec}."
          )
        host_cardinalities = np.asarray(
            multihost_utils.process_allgather(post_cardinality)
        ).reshape(-1)
        if np.unique(host_cardinalities).size > 1:
          raise ValueError(
              "DropRemainder.PAD_WITH_MASK across multiple JAX processes"
              " requires equal dataset cardinality across hosts after"
              " `post_batch_transforms`, but got per-host cardinalities"
              f" {host_cardinalities.tolist()}."
          )
        if not any_allow_drop and (
            not isinstance(ds.element_spec, Mapping)
            or PADDING_MASK_KEY not in ds.element_spec
        ):
          raise ValueError(
              "DropRemainder.PAD_WITH_MASK across multiple JAX processes"
              f" requires {PADDING_MASK_KEY!r} in `element_spec` after"
              " `post_batch_transforms` (unless explicitly dropped or"
              f" renamed), but got element_spec={ds.element_spec}."
          )

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


def _normalize_transforms(
    transforms: _Transforms,
) -> list[tr_utils.Transformation]:
  if isinstance(transforms, Mapping):
    return list(transforms.values())
  return list(transforms)


def _has_batched_index(ds: tf.data.Dataset) -> bool:
  """Returns whether `ds` carries a non-empty `grain.INDEX` with rank >= 1."""
  if not isinstance(ds.element_spec, Mapping):
    return False
  index_spec = ds.element_spec.get(grain.INDEX)
  if not isinstance(index_spec, tf.TensorSpec):
    return False
  rank = index_spec.shape.rank
  return (
      rank is not None
      and rank >= 1
      and tf.compat.dimension_value(index_spec.shape[0]) != 0
  )


def _fold_in_tensor(seed: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
  """Deterministically folds an arbitrary integer tensor into a `[2]` TF seed."""
  seed = tf.ensure_shape(tf.cast(seed, tf.int32), [2])
  flat = tf.reshape(values, [-1])

  def _fold_non_empty() -> tf.Tensor:
    fp_bytes = tf.fingerprint(tf.expand_dims(flat, axis=0))
    words = tf.bitcast(tf.reshape(fp_bytes, [2, 4]), tf.int32)
    out = tf.random.experimental.stateless_fold_in(seed, words[0])
    out = tf.random.experimental.stateless_fold_in(out, words[1])
    out = tf.random.experimental.stateless_split(out, num=1)[0]
    return tf.ensure_shape(out, [2])

  return tf.cond(
      tf.greater(tf.size(flat), 0),
      _fold_non_empty,
      lambda: seed,
  )


def _derive_post_batch_seed(
    base_seed: tuple[int, int],
    *,
    index: tf.Tensor | None,
    raw_seed: tf.Tensor | None,
) -> tf.Tensor:
  """Derives a stateless `[2]` int32 seed for a post-batch transform."""
  seed = tf.constant(base_seed, dtype=tf.int32)
  if index is not None:
    seed = _fold_in_tensor(seed, index)
  if raw_seed is not None:
    flat_seed = tf.reshape(raw_seed, [-1])
    non_zero_seed = tf.boolean_mask(flat_seed, tf.not_equal(flat_seed, 0))
    seed = _fold_in_tensor(seed, non_zero_seed)
  return seed


@dataclasses.dataclass(frozen=True)
class _PostBatchRandomMapTransform(grain.MapTransform):
  """Stateless `MapTransform` wrapper for post-batch `RandomMapTransform`."""

  transform: grain.RandomMapTransform
  base_seed: tuple[int, int]

  def name(self) -> str:
    attr = self.transform.name
    return attr() if callable(attr) else attr

  def num_parallel_calls_hint(self) -> int | None:
    attr = self.transform.num_parallel_calls_hint
    return attr() if callable(attr) else attr

  def map(self, element: Any) -> Any:
    if not isinstance(element, Mapping):
      raise ValueError(
          "Grain expects dataset elements to be dictionaries but"
          f" transformation {self.transform} got {element}."
      )
    features = dict(element)
    has_seed = grain.SEED in features
    raw_seed = features.pop(grain.SEED, None)
    seed = _derive_post_batch_seed(
        self.base_seed,
        index=features.get(grain.INDEX),
        raw_seed=raw_seed,
    )

    out = self.transform.random_map(features, seed)
    if has_seed and isinstance(out, Mapping):
      out = {**out, grain.SEED: raw_seed}
    return out


@dataclasses.dataclass(frozen=True)
class _PostBatchPreprocessFnTransform(grain.MapTransform):
  """Stateless `MapTransform` wrapper for post-batch CLU `PreprocessFn`."""

  transform: preprocess_spec.PreprocessFn | preprocess_spec.RandomMapTransform
  base_seed: tuple[int, int]

  def name(self) -> str:
    return type(self.transform).__name__

  def map(self, element: Any) -> Any:
    if not isinstance(element, Mapping):
      raise ValueError(
          "Grain expects dataset elements to be dictionaries but"
          f" transformation {self.transform} got {element}."
      )
    meta_features, ex_features = grain_utils.split_grain_meta_features(
        dict(element)
    )
    if not isinstance(ex_features, Mapping):
      raise ValueError(
          "CLU preprocessing expects dictionary features, got"
          f" {ex_features!r} for {self.transform}."
      )
    seed = _derive_post_batch_seed(
        self.base_seed,
        index=meta_features.get(grain.INDEX),
        raw_seed=meta_features.get(grain.SEED),
    )
    ex_input = dict(ex_features)
    ex_input[grain.SEED] = seed
    out = self.transform(ex_input)
    if isinstance(out, Mapping):
      out = {k: v for k, v in out.items() if k != grain.SEED}
    return grain_utils.merge_grain_meta_features(meta_features, out)


def _validate_padding_mask_output(out: Mapping[str, Any]) -> Mapping[str, Any]:
  """Validates `out[PADDING_MASK_KEY]` dtype, rank, and batch dimension."""
  raw_mask = out[PADDING_MASK_KEY]
  if raw_mask is None or isinstance(
      raw_mask, (tf.RaggedTensor, tf.SparseTensor)
  ):
    raise ValueError(
        f"{PADDING_MASK_KEY!r} must be a dense 1D tf.bool tensor, got"
        f" {raw_mask!r}."
    )
  mask = tf.convert_to_tensor(raw_mask)
  if mask.dtype != tf.bool:
    raise ValueError(
        f"{PADDING_MASK_KEY!r} must have dtype tf.bool, got {mask.dtype}."
    )

  assert_ops: list[tf.Operation] = []
  if mask.shape.rank is not None:
    if mask.shape.rank != 1:
      raise ValueError(
          f"{PADDING_MASK_KEY!r} must have rank 1, got shape {mask.shape}."
      )
    mask_static_dim = tf.compat.dimension_value(mask.shape[0])
    if mask_static_dim is not None and mask_static_dim <= 0:
      raise ValueError(
          f"{PADDING_MASK_KEY!r} must have positive batch size, got shape"
          f" {mask.shape}."
      )
  else:
    mask_static_dim = None
    assert_ops.append(
        tf.debugging.assert_rank(
            mask,
            1,
            message=f"{PADDING_MASK_KEY!r} must have rank 1.",
        )
    )

  with tf.control_dependencies(assert_ops):
    mask_batch_dim = tf.shape(mask, out_type=tf.int32)[0]
  if mask_static_dim is None:
    assert_ops.append(
        tf.debugging.assert_greater(
            mask_batch_dim,
            0,
            message=f"{PADDING_MASK_KEY!r} must have positive batch size.",
        )
    )

  ex_features = {
      k: v
      for k, v in out.items()
      if k not in grain.META_FEATURES and k != PADDING_MASK_KEY
  }
  expected_static_dim: int | None = mask_static_dim
  for leaf in grain_utils.non_none_leaves(ex_features):
    if not isinstance(leaf, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
      leaf = tf.convert_to_tensor(leaf)
    if leaf.shape.rank is not None:
      if leaf.shape.rank < 1:
        raise ValueError(
            "Output feature leaf must have rank >= 1 to match"
            f" {PADDING_MASK_KEY!r} of shape {mask.shape}, got {leaf.shape}."
        )
      leaf_static_dim = tf.compat.dimension_value(leaf.shape[0])
      if leaf_static_dim is not None:
        if leaf_static_dim <= 0:
          raise ValueError(
              "Output feature leaf must have positive batch size, got"
              f" {leaf.shape}."
          )
        if expected_static_dim is None:
          expected_static_dim = leaf_static_dim
        elif leaf_static_dim != expected_static_dim:
          raise ValueError(
              f"{PADDING_MASK_KEY!r} batch size ({expected_static_dim}) does"
              f" not match output feature batch size ({leaf_static_dim})."
          )
    else:
      leaf_static_dim = None

    if mask_static_dim is None or leaf_static_dim is None:
      leaf_batch_dim, leaf_ops = grain_utils.leaf_batch_size(leaf)
      assert_ops.extend(leaf_ops)
      assert_ops.append(
          tf.debugging.assert_equal(
              leaf_batch_dim,
              mask_batch_dim,
              message=(
                  f"{PADDING_MASK_KEY!r} batch size does not match output"
                  " feature batch size."
              ),
          )
      )

  if assert_ops:
    return grain_utils.attach_control_dependencies(
        dict(out) | {PADDING_MASK_KEY: mask}, assert_ops
    )
  if isinstance(raw_mask, tf.Tensor):
    return out
  return dict(out) | {PADDING_MASK_KEY: mask}


def _validate_padding_mask_spec(spec: Any) -> None:
  """Validates `spec[PADDING_MASK_KEY]` statically on a dataset element spec."""
  if not isinstance(spec, Mapping) or PADDING_MASK_KEY not in spec:
    return
  mask_spec = spec[PADDING_MASK_KEY]
  if not isinstance(mask_spec, tf.TensorSpec) or mask_spec.dtype != tf.bool:
    raise ValueError(
        f"{PADDING_MASK_KEY!r} must be a tf.bool TensorSpec, got {mask_spec!r}"
        f" in element_spec={spec}."
    )
  if mask_spec.shape.rank is not None:
    if mask_spec.shape.rank != 1:
      raise ValueError(
          f"{PADDING_MASK_KEY!r} must have rank 1, got {mask_spec!r} in"
          f" element_spec={spec}."
      )
    mask_dim = tf.compat.dimension_value(mask_spec.shape[0])
    if mask_dim is not None and mask_dim <= 0:
      raise ValueError(
          f"{PADDING_MASK_KEY!r} must have positive batch size, got"
          f" {mask_spec!r} in element_spec={spec}."
      )
  else:
    mask_dim = None

  ex_spec = {
      k: v
      for k, v in spec.items()
      if k not in grain.META_FEATURES and k != PADDING_MASK_KEY
  }
  expected_dim: int | None = mask_dim
  for leaf in grain_utils.non_none_leaves(ex_spec):
    if not isinstance(
        leaf, (tf.TensorSpec, tf.RaggedTensorSpec, tf.SparseTensorSpec)
    ):
      continue
    if leaf.shape.rank is not None:
      if leaf.shape.rank < 1:
        raise ValueError(
            "Output feature leaf must have rank >= 1 when"
            f" {PADDING_MASK_KEY!r} is present, got {leaf!r} in"
            f" element_spec={spec}."
        )
      leaf_dim = tf.compat.dimension_value(leaf.shape[0])
      if leaf_dim is not None:
        if leaf_dim <= 0:
          raise ValueError(
              f"Output feature leaf must have positive batch size, got {leaf!r}"
              f" in element_spec={spec}."
          )
        if expected_dim is None:
          expected_dim = leaf_dim
        elif leaf_dim != expected_dim:
          raise ValueError(
              f"{PADDING_MASK_KEY!r} batch size ({expected_dim}) does not match"
              f" output feature batch size ({leaf_dim}) in"
              f" element_spec={spec}."
          )


@dataclasses.dataclass(frozen=True)
class _PreservePaddingMaskMapTransform(grain.MapTransform):
  """Preserves `PADDING_MASK_KEY` across post-batch map transforms."""

  transform: grain.MapTransform
  allow_drop: bool = False

  def name(self) -> str:
    attr = self.transform.name
    return attr() if callable(attr) else attr

  def num_parallel_calls_hint(self) -> int | None:
    attr = self.transform.num_parallel_calls_hint
    return attr() if callable(attr) else attr

  def map(self, element: Any) -> Any:
    has_mask = isinstance(element, Mapping) and PADDING_MASK_KEY in element
    padding_mask = element[PADDING_MASK_KEY] if has_mask else None
    out = self.transform.map(element)
    if (
        has_mask
        and not self.allow_drop
        and isinstance(out, Mapping)
        and grain.RECORD not in out
        and PADDING_MASK_KEY not in out
    ):
      out = {**out, PADDING_MASK_KEY: padding_mask}
    if isinstance(out, Mapping) and PADDING_MASK_KEY in out:
      out = _validate_padding_mask_output(out)
    return out


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
    has_static_shapes = all(
        spec.shape.is_fully_defined()
        for spec in jax.tree.leaves(ds.element_spec)
    )
    if self.add_padding_mask and cardinality >= 0 and has_static_shapes:
      # When dataset cardinality is statically known, all element shapes are
      # fully defined, and `add_padding_mask=True`, synchronize local shard
      # lengths across JAX processes and pad the unbatched dataset up to a
      # common multiple of `batch_size`. This guarantees that every host
      # produces the exact same number of batches even when sharding yields
      # uneven per-host dataset lengths. Note that `process_allgather` is a
      # collective barrier executed once when the cached `_root_ds` is built,
      # so all JAX processes must build the pipeline together.
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
            lambda features: dict(features) | {
                self.mask_key: tf.constant(False, dtype=tf.bool)
            },
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

    if self.add_padding_mask and jax.process_count() > 1:
      raise ValueError(
          "DropRemainder.PAD_WITH_MASK across multiple JAX processes requires "
          "known dataset cardinality and fully defined static element shapes "
          "to equalize batch counts across hosts, but got cardinality="
          f"{cardinality} and element_spec={ds.element_spec}."
      )

    # Fallback path when `add_padding_mask=False` (`DropRemainder.PAD`), when
    # dataset cardinality is unknown/infinite (e.g. after `.filter()`), or when
    # any element shape is not fully defined (e.g. after `tf.py_function`):
    # batch with `drop_remainder=False` and dynamically pad any partial batch to
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
      is_padded = tf.range(self.batch_size) >= valid_batch_size
      if self.add_padding_mask:
        # Mark indices `>= valid_batch_size` as padded (`True`), OR-ing with any
        # pre-existing `mask_key` tensor in the batch.
        if self.mask_key in batch:
          padded_batch[self.mask_key] = tf.logical_or(
              padded_batch[self.mask_key], is_padded
          )
        else:
          padded_batch[self.mask_key] = is_padded
        padded_batch[self.mask_key].set_shape((self.batch_size,))
      # Replace padded `grain.INDEX` entries with `-1` in both `PAD` and
      # `PAD_WITH_MASK` modes so padded slots do not collide with record 0.
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
    # Check `isinstance(leaf, TfArray)` and `leaf.shape.rank is not None`
    # before `TfArray["_ *_"]` because `None` leaves have no `.shape` and
    # `TfArray["_ *_"]` calls `tuple(leaf.shape)`, which raises `ValueError`
    # on unranked `tf.Tensor`s (e.g. after `tf.py_function`).
    if (
        isinstance(leaf, TfArray)
        and leaf.shape.rank is not None
        and isinstance(leaf, TfArray["_ *_"])
    ):
      return leaf
  raise ValueError(
      f"Could not find any tensor leaf with rank >= 1 in batch: {batch}"
  )
