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

from collections.abc import Mapping
import dataclasses
import functools
from typing import Any

import jax
from kauldron import random
from kauldron.data.tf import base
from kauldron.data.tf import grain_utils
from kauldron.ktyping import PRNGKey
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
  weights: None | list[float | int] = None
  stop_on_empty_dataset: bool = False
  rerandomize_each_iteration: bool = True

  def __post_init__(self):
    super().__post_init__()
    if not all(isinstance(ds, base.TFDataPipeline) for ds in self.datasets):
      raise ValueError(
          'All datasets in `SampleFromDatasets` should inherit from'
          ' `TFDataPipeline`.'
      )

  def ds_for_current_process(self, rng: PRNGKey) -> tf.data.Dataset:
    # We make sure each nested dataset get a different seed, and propagate
    # `batch_drop_remainder` to unbatched child datasets when the mixture
    # itself performs batching so sharding matches the mixture's remainder mode.
    datasets = []
    for i, ds in enumerate(self.datasets):
      if self.batch_size and ds.batch_size is None:
        ds = dataclasses.replace(
            ds, batch_drop_remainder=self.batch_drop_remainder
        )
      child_ds = ds.ds_with_transforms(jax.random.fold_in(rng, i))
      datasets.append(child_ds)

    ds = tf.data.Dataset.sample_from_datasets(
        datasets,
        weights=self.weights,
        seed=random.random_seed(
            random.fold_in_str(rng, 'sample_from_datasets')
        ),
        stop_on_empty_dataset=self.stop_on_empty_dataset,
        rerandomize_each_iteration=self.rerandomize_each_iteration,
    )
    return ds

  @functools.cached_property
  def _supports_symbolic_checkpoint(self) -> bool:
    return all(ds._supports_symbolic_checkpoint for ds in self.datasets)  # pylint: disable=protected-access


def _maybe_hoist_child_padding_masks(ds: tf.data.Dataset) -> tf.data.Dataset:
  """Pops child `__is_padding__` masks and hoists their OR to the top level."""
  spec = ds.element_spec
  if not isinstance(spec, Mapping):
    return ds

  child_mask_specs: dict[str, Any] = {}
  for ds_name, child_spec in spec.items():
    if isinstance(child_spec, Mapping) and base.PADDING_MASK_KEY in child_spec:
      child_mask_specs[ds_name] = child_spec[base.PADDING_MASK_KEY]

  if not child_mask_specs:
    return ds

  if base.PADDING_MASK_KEY in spec:
    raise ValueError(
        f'Cannot hoist child {base.PADDING_MASK_KEY!r} masks because'
        f' ZipDatasets already has a dataset key {base.PADDING_MASK_KEY!r}.'
    )

  static_mask_dim: int | None = None
  for ds_name, mask_spec in child_mask_specs.items():
    if not isinstance(mask_spec, tf.TensorSpec) or mask_spec.dtype != tf.bool:
      raise ValueError(
          f'Child dataset {ds_name!r} has invalid {base.PADDING_MASK_KEY!r}'
          f' spec {mask_spec!r}; expected 1D tf.bool TensorSpec.'
      )
    if mask_spec.shape.rank is not None:
      if mask_spec.shape.rank != 1:
        raise ValueError(
            f'Child dataset {ds_name!r} {base.PADDING_MASK_KEY!r} must have'
            f' rank 1, got {mask_spec!r}.'
        )
      dim0 = tf.compat.dimension_value(mask_spec.shape[0])
      if dim0 is not None:
        if dim0 <= 0:
          raise ValueError(
              f'Child dataset {ds_name!r} {base.PADDING_MASK_KEY!r} must have'
              f' positive batch size, got {mask_spec!r}.'
          )
        if static_mask_dim is None:
          static_mask_dim = dim0
        elif dim0 != static_mask_dim:
          raise ValueError(
              'Child datasets in pre-batched `ZipDatasets` have mismatched'
              f' {base.PADDING_MASK_KEY!r} batch sizes: {static_mask_dim} vs'
              f' {dim0}.'
          )

  def _hoist_masks(element: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    masks: list[tf.Tensor] = []
    assert_ops: list[tf.Operation] = []
    for ds_name, child_ex in element.items():
      if isinstance(child_ex, Mapping) and base.PADDING_MASK_KEY in child_ex:
        child_copy = dict(child_ex)
        mask = tf.convert_to_tensor(child_copy.pop(base.PADDING_MASK_KEY))
        if mask.shape.rank is None:
          assert_ops.append(
              tf.debugging.assert_rank(
                  mask,
                  1,
                  message=(
                      f'Child dataset {ds_name!r} {base.PADDING_MASK_KEY!r}'
                      ' must have rank 1.'
                  ),
              )
          )
        masks.append(mask)
        out[ds_name] = child_copy
      else:
        out[ds_name] = child_ex

    with tf.control_dependencies(assert_ops):
      combined = masks[0]
      first_len = tf.shape(combined, out_type=tf.int32)[0]
      len_asserts: list[tf.Operation] = []
      for other_mask in masks[1:]:
        if (
            tf.compat.dimension_value(combined.shape[0]) is None
            or tf.compat.dimension_value(other_mask.shape[0]) is None
        ):
          len_asserts.append(
              tf.debugging.assert_equal(
                  tf.shape(other_mask, out_type=tf.int32)[0],
                  first_len,
                  message=(
                      'Child datasets in pre-batched `ZipDatasets` must have'
                      f' matching {base.PADDING_MASK_KEY!r} batch sizes.'
                  ),
              )
          )
      with tf.control_dependencies(len_asserts):
        for other_mask in masks[1:]:
          combined = tf.math.logical_or(combined, other_mask)
        if static_mask_dim is not None:
          combined.set_shape((static_mask_dim,))
        out[base.PADDING_MASK_KEY] = combined

    return grain_utils.attach_control_dependencies(
        out, [*assert_ops, *len_asserts]
    )

  return ds.map(_hoist_masks)


@dataclasses.dataclass(frozen=True)
class ZipDatasets(base.TFDataPipeline):
  """Creates a Dataset by zipping together the given datasets.

  Attributes:
    datasets: Dictionary of datasets to sample from.
  """

  datasets: dict[str, base.TFDataPipeline]
  _: dataclasses.KW_ONLY

  def __post_init__(self):
    super().__post_init__()
    if not all(
        isinstance(ds, base.TFDataPipeline) for ds in [*self.datasets.values()]
    ):
      raise ValueError(
          'All datasets in `ZipDatasets` should inherit from `TFDataPipeline`.'
      )

  def ds_for_current_process(self, rng: PRNGKey) -> tf.data.Dataset:
    datasets = {}
    children_are_batched = bool(self.datasets)
    for i, (ds_name, ds) in enumerate(self.datasets.items()):
      if self.batch_size and ds.batch_size is None:
        ds = dataclasses.replace(
            ds, batch_drop_remainder=self.batch_drop_remainder
        )
      child_ds = ds.ds_with_transforms(jax.random.fold_in(rng, i))
      children_are_batched = (
          children_are_batched
          and ds._is_batched_ds(child_ds)  # pylint: disable=protected-access
      )
      datasets[ds_name] = child_ds
    object.__setattr__(self, '_children_are_batched', children_are_batched)

    # Drop grain meta features
    datasets = {
        ds_name: ds.map(lambda ex: grain_utils.split_grain_meta_features(ex)[1])
        for ds_name, ds in datasets.items()
    }

    ds = tf.data.Dataset.zip(datasets)
    if children_are_batched and not self.batch_size:
      ds = _maybe_hoist_child_padding_masks(ds)
    return ds

  def _is_prebatched_source(self, ds: tf.data.Dataset) -> bool:
    return (
        not self.batch_size
        and getattr(self, '_children_are_batched', False)
        and grain_utils.has_prebatched_spec(
            ds.element_spec, allow_unknown_rank=True
        )
    )

  @functools.cached_property
  def _supports_symbolic_checkpoint(self) -> bool:
    return all(
        ds._supports_symbolic_checkpoint for ds in self.datasets.values()  # pylint: disable=protected-access
    )
