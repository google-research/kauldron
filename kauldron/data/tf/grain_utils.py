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

"""Grain utils to abstract the meta features."""

from collections.abc import Mapping
import sys
from typing import Any

from etils import epy
from etils.etree import nest as etree  # pylint: disable=g-importing-member
import grain.tensorflow as grain
import jax
from kauldron import random
from kauldron.ktyping import PRNGKey
import tensorflow as tf

_SUPPORTED_TENSOR_SPECS = (
    tf.TensorSpec,
    tf.RaggedTensorSpec,
    tf.SparseTensorSpec,
)


def _unexpected_example_structure(ex: Any) -> Exception:
  """Drop grain meta features."""
  spec = epy.pretty_repr(etree.spec_like(ex))
  return ValueError(f"Unexpected example structure: {spec}")


def split_grain_meta_features(
    features: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
  """Extract the non-grain features."""
  if not isinstance(features, dict) or grain.INDEX not in features:
    raise _unexpected_example_structure(features)

  # b/326590491: Grain meta features are inconsitent. Either:
  # * `{**meta, **X}`
  # * `{**meta, '_record': X}`
  if grain.RECORD in features:
    if any(not k.startswith("_") for k in features):
      raise _unexpected_example_structure(features)
    ex_features = features.pop(grain.RECORD)
    meta_features = features
  else:  # Dict and keys merged
    ex_features, meta_features = epy.splitby(
        features.items(),
        predicate=lambda key_and_value: key_and_value[0] in grain.META_FEATURES,
    )
    meta_features = dict(meta_features)
    ex_features = dict(ex_features)
  return meta_features, ex_features


def merge_grain_meta_features(meta_features, ex_features):
  if not isinstance(ex_features, dict):
    ex_features = {grain.RECORD: ex_features}
  return meta_features | ex_features


def _is_none_spec(leaf: Any) -> bool:
  return leaf is None or (
      isinstance(leaf, tf.TypeSpec) and leaf.value_type is type(None)
  )


def non_none_leaves(tree: Any) -> list[Any]:
  """Returns all non-`None` leaves in `tree`."""
  return [leaf for leaf in jax.tree.leaves(tree) if not _is_none_spec(leaf)]


def _extract_example_spec(spec: Any) -> Any:
  """Strips `grain.META_FEATURES` from `spec` when present."""
  if isinstance(spec, Mapping):
    if grain.RECORD in spec and all(k.startswith("_") for k in spec):
      return spec[grain.RECORD]
    return {k: v for k, v in spec.items() if k not in grain.META_FEATURES}
  return spec


def validate_prebatched_spec(
    spec: Any,
    *,
    allow_unknown_rank: bool = True,
) -> int | None:
  """Validates a pre-batched element spec and returns static batch size if known."""
  ex_spec = _extract_example_spec(spec)
  leaves = non_none_leaves(ex_spec)
  if not leaves:
    raise ValueError(
        "Pre-batched dataset must contain at least one non-None tensor leaf,"
        f" got element_spec={spec}."
    )

  static_batch_size: int | None = None
  for leaf in leaves:
    if not isinstance(leaf, _SUPPORTED_TENSOR_SPECS):
      raise ValueError(
          "Pre-batched dataset leaves must be TensorSpec, RaggedTensorSpec,"
          f" or SparseTensorSpec, got {leaf!r} in element_spec={spec}."
      )
    rank = leaf.shape.rank
    if rank is None:
      if not allow_unknown_rank:
        raise ValueError(
            "Pre-batched dataset leaf has unknown static rank:"
            f" {leaf!r} in element_spec={spec}."
        )
      continue
    if rank < 1:
      raise ValueError(
          "All non-None leaves in a pre-batched dataset must have rank >= 1,"
          f" got {leaf!r} in element_spec={spec}."
      )
    dim0 = tf.compat.dimension_value(leaf.shape[0])
    if dim0 is not None:
      if dim0 <= 0:
        raise ValueError(
            "Pre-batched dataset leaves must have non-zero leading batch"
            f" dimension, got {leaf!r} in element_spec={spec}."
        )
      if static_batch_size is None:
        static_batch_size = dim0
      elif dim0 != static_batch_size:
        raise ValueError(
            "All leaves in a pre-batched dataset must have matching leading"
            f" batch dimensions, got {static_batch_size} vs {dim0} in"
            f" element_spec={spec}."
        )
  return static_batch_size


def has_prebatched_spec(
    spec: Any,
    *,
    allow_unknown_rank: bool = False,
) -> bool:
  """Returns whether all non-`None` leaves in `spec` have compatible batch dims."""
  try:
    validate_prebatched_spec(spec, allow_unknown_rank=allow_unknown_rank)
  except ValueError:
    return False
  return True


def leaf_batch_size(leaf: Any) -> tuple[tf.Tensor, list[tf.Operation]]:
  """Returns `(batch_size_int32, assert_ops)` for a tensor/ragged/sparse leaf."""
  if isinstance(leaf, tf.RaggedTensor):
    return tf.cast(leaf.nrows(), tf.int32), []
  if isinstance(leaf, tf.SparseTensor):
    if leaf.shape.rank is not None and leaf.shape.rank >= 1:
      return tf.cast(leaf.dense_shape[0], tf.int32), []
    assert_op = tf.debugging.assert_greater_equal(
        tf.size(leaf.dense_shape),
        1,
        message="Expected batched SparseTensor leaf with rank >= 1.",
    )
    with tf.control_dependencies([assert_op]):
      return tf.cast(leaf.dense_shape[0], tf.int32), [assert_op]

  tensor = tf.convert_to_tensor(leaf)
  if tensor.shape.rank is not None:
    if tensor.shape.rank < 1:
      raise ValueError(
          f"Expected batched tensor leaf with rank >= 1, got {tensor.shape}."
      )
    return tf.shape(tensor, out_type=tf.int32)[0], []

  assert_op = tf.debugging.assert_rank_at_least(
      tensor,
      1,
      message="Expected batched tensor leaf with rank >= 1.",
  )
  with tf.control_dependencies([assert_op]):
    return tf.shape(tensor, out_type=tf.int32)[0], [assert_op]


def _identity_leaf(leaf: Any) -> Any:
  """Wraps tensor, ragged, or sparse leaves in `tf.identity` ops."""
  if leaf is None:
    return None
  if isinstance(leaf, tf.RaggedTensor):
    return tf.RaggedTensor.from_nested_row_splits(
        tf.identity(leaf.flat_values),
        tuple(tf.identity(s) for s in leaf.nested_row_splits),
        validate=False,
    )
  if isinstance(leaf, tf.SparseTensor):
    return tf.SparseTensor(
        indices=tf.identity(leaf.indices),
        values=tf.identity(leaf.values),
        dense_shape=tf.identity(leaf.dense_shape),
    )
  return tf.identity(leaf)


def attach_control_dependencies(
    tree: Any, assert_ops: list[tf.Operation]
) -> Any:
  """Attaches `assert_ops` as control dependencies to every leaf in `tree`."""
  if not assert_ops:
    return tree
  with tf.control_dependencies(assert_ops):
    return jax.tree.map(_identity_leaf, tree)


def maybe_add_grain_meta_features(
    ds: tf.data.Dataset,
    *,
    rng: PRNGKey,
    is_prebatched: bool = False,
) -> tf.data.Dataset:
  """Add grain meta features."""
  # Dataset already has grain meta features.
  if isinstance(ds.element_spec, Mapping) and grain.INDEX in ds.element_spec:
    if is_prebatched:
      static_batch_size = validate_prebatched_spec(
          ds.element_spec, allow_unknown_rank=True
      )
      index_spec = ds.element_spec[grain.INDEX]
      if (
          not isinstance(index_spec, tf.TensorSpec)
          or index_spec.shape.rank is None
          or index_spec.shape.rank < 1
      ):
        raise ValueError(
            "`is_prebatched=True` requires batched elements with rank >= 1,"
            f" got {grain.INDEX}={index_spec!r} in"
            f" element_spec={ds.element_spec}."
        )
      index_dim0 = tf.compat.dimension_value(index_spec.shape[0])
      if index_dim0 == 0 or (
          index_dim0 is not None
          and static_batch_size is not None
          and index_dim0 != static_batch_size
      ):
        raise ValueError(
            f"`is_prebatched=True` got incompatible {grain.INDEX} shape"
            f" {index_spec.shape} for element_spec={ds.element_spec}."
        )
    return ds

  if is_prebatched:
    static_batch_size = validate_prebatched_spec(
        ds.element_spec, allow_unknown_rank=True
    )
  else:
    static_batch_size = None

  # This should be deterministic as long as the `ds` is deterministic.
  sampler = grain.TfDefaultIndexSampler(
      num_records=sys.maxsize,  # Infinite iterator
      shuffle=False,
      seed=random.random_seed(random.fold_in_str(rng, "grain_metadata")),
      shard_options=grain.ShardByJaxProcess(),
      num_epochs=None,
  )
  index_ds = sampler.get_index_dataset(grain.FirstIndex())
  ds = tf.data.Dataset.zip(index_ds, ds)
  if not is_prebatched:
    return ds.map(merge_grain_meta_features)

  def _merge_prebatched(meta_features, ex_features):
    leaves = non_none_leaves(ex_features)
    assert_ops: list[tf.Operation] = []
    batch_dims: list[tf.Tensor] = []
    for leaf in leaves:
      dim, ops = leaf_batch_size(leaf)
      batch_dims.append(dim)
      assert_ops.extend(ops)

    if static_batch_size is not None:
      batch_size = tf.constant(static_batch_size, dtype=tf.int32)
    else:
      batch_size = batch_dims[0]
      assert_ops.append(
          tf.debugging.assert_greater(
              batch_size,
              0,
              message=(
                  "Pre-batched dataset leaves must have non-zero leading batch"
                  " dimension."
              ),
          )
      )

    for leaf, dim in zip(leaves, batch_dims, strict=True):
      leaf_static_dim0 = (
          tf.compat.dimension_value(leaf.shape[0])
          if leaf.shape.rank is not None and leaf.shape.rank >= 1
          else None
      )
      if leaf_static_dim0 is None:
        assert_ops.append(
            tf.debugging.assert_equal(
                dim,
                batch_size,
                message=(
                    "All leaves in a pre-batched dataset must have matching"
                    " leading batch dimensions."
                ),
            )
        )

    with tf.control_dependencies(assert_ops):
      batch_size_i64 = tf.cast(batch_size, tf.int64)
      offsets = tf.range(batch_size_i64, dtype=tf.int64)
      expanded_meta: dict[str, tf.Tensor] = {}
      for key, value in meta_features.items():
        if key in (grain.INDEX, grain.RECORD_KEY):
          expanded = value * batch_size_i64 + offsets
          if static_batch_size is not None:
            expanded.set_shape((static_batch_size,))
        elif key == grain.SEED:
          num_splits = (
              static_batch_size if static_batch_size is not None else batch_size
          )
          expanded = tf.random.experimental.stateless_split(
              value, num=num_splits
          )
          if static_batch_size is not None:
            expanded.set_shape((static_batch_size, 2))
        else:
          expanded = tf.fill([batch_size], value)
          if static_batch_size is not None:
            expanded.set_shape((static_batch_size,))
        expanded_meta[key] = expanded

    ex_features = attach_control_dependencies(ex_features, assert_ops)
    return merge_grain_meta_features(expanded_meta, ex_features)

  return ds.map(_merge_prebatched)
