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

"""Utils for using Kauldron transforms with tfgrain."""

import functools
from typing import Any, Callable, Mapping

from grain._src.tensorflow import transforms as grain_transforms
import grain.tensorflow as grain
from kauldron.data.tf import grain_utils
from kauldron.data.transforms import abc as tr_abc
from kauldron.data.transforms import normalize as tr_normalize
import tensorflow as tf


class TfGrainMapAdapter(tr_normalize.TransformAdapter, grain.MapTransform):
  """Adapter for `kd.data.MapTransform` to tfgrain."""

  @property
  def name(self):
    """Forward the name of this transformation (if any), to aid in debugging."""
    # Used by tfgrain to name the operations in the tf graph.
    return getattr(self.transform, 'name', getattr(super(), 'name'))

  @property
  def num_parallel_calls_hint(self):
    """Forward the num_parallel_calls_hint of this transformation (if any)."""
    # Can be used to modify the default parallelization behavior of tfgrain.
    return getattr(
        self.transform,
        'num_parallel_calls_hint',
        getattr(super(), 'num_parallel_calls_hint'),
    )

  def map(self, element: Any) -> Any:
    # Required due to b/326590491.
    meta_features, ex_features = grain_utils.split_grain_meta_features(element)
    out = self.transform.map(ex_features)
    return grain_utils.merge_grain_meta_features(meta_features, out)


class TfGrainCallableAdapter(tr_normalize.TransformAdapter, grain.MapTransform):
  """Adapter for any callable to a tfgrain MapTransform."""

  def map(self, element: Any) -> Any:
    # Required due to b/326590491.
    meta_features, ex_features = grain_utils.split_grain_meta_features(element)
    out = self.transform(ex_features)
    return grain_utils.merge_grain_meta_features(meta_features, out)


class TfGrainFilterAdapter(
    tr_normalize.TransformAdapter, grain.FilterTransform
):
  """Adapter from `kd.data.FilterTransform` to tfgrain."""

  @property
  def name(self):
    """Forward the name of this transformation (if any), to aid in debugging."""
    # Used by tfgrain to name the operations in the tf graph.
    return getattr(self.transform, 'name', getattr(super(), 'name'))

  def filter(self, elements: Any) -> Any:
    # Required due to b/326590491.
    _, ex_features = grain_utils.split_grain_meta_features(elements)
    return self.transform.filter(ex_features)


_KD_TO_TFGRAIN_ADAPTERS = {
    tr_abc.MapTransform: TfGrainMapAdapter,
    tr_abc.FilterTransform: TfGrainFilterAdapter,
    Callable: TfGrainCallableAdapter,  # support grand-vision preprocessing ops
}


def _adapt_for_tfgrain(
    transform: tr_normalize.Transformation,
) -> grain.Transformation:
  if isinstance(transform, grain.Transformation):
    return transform
  return tr_normalize.adapt_transform(transform, _KD_TO_TFGRAIN_ADAPTERS)


def apply_transformations(
    ds: tf.data.Dataset,
    transforms: tr_normalize.Transformations,
) -> tf.data.Dataset:
  """Wrapper around grain to apply the transformations."""
  if isinstance(transforms, Mapping):
    transforms = transforms.values()
  transforms = [_adapt_for_tfgrain(tr) for tr in transforms]
  return grain_transforms.apply_transformations(ds, transforms, strict=True)


# TODO(b/279722981): Pytype do not bind methods correctly when calling
# `random_map = wrap_map(random_map)` on `ElementWiseRandomTransform`, so
# return `Any` instead. Annotate here rather than on
# `ElementWiseRandomTransform` as annotation interfere with `@dataclass`.
def wrap_map(fn, *, restore_meta_features: bool = True) -> Any:
  """Wrap the transform function to remove the grain internals.

  Required due to b/326590491.

  Args:
    fn: The function to wrap
    restore_meta_features: If True, restore the grain meta features on the
      output

  Returns:
    The decorated function.
  """

  @functools.wraps(fn)
  def new_map(self, element: dict[str, Any], *args, **kwargs) -> dict[str, Any]:
    meta_features, ex_features = grain_utils.split_grain_meta_features(element)
    out = fn(self, ex_features, *args, **kwargs)
    if restore_meta_features:
      return grain_utils.merge_grain_meta_features(meta_features, out)
    else:
      return out

  return new_map


def get_target_shape(t: tf.Tensor, target_shape):
  """Resolve the `dynamic` portions of `target_shape`."""
  finale_shape = []
  dynamic_shape = tf.shape(t)
  for i, (static_dim, target_dim) in enumerate(zip(t.shape, target_shape)):
    if target_dim is not None:
      finale_shape.append(target_dim)
    elif static_dim is not None:
      finale_shape.append(static_dim)
    else:
      finale_shape.append(dynamic_shape[i])
  return finale_shape
