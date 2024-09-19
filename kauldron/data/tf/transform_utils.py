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

"""Transform utils."""

import functools
from typing import Any, TypeVar

from grain import tensorflow as grain
from grain._src.tensorflow import transforms as grain_transforms
from kauldron.data.tf import grain_utils
from kauldron.data.transforms import abc as tr_abc
from kauldron.data.transforms import normalize as tr_normalize
import tensorflow as tf

_FnT = TypeVar('_FnT')

# Kauldron transforms for kd.data.tf supports both `kd.data.MapTransform` and
# `tfgrain.MapTransform`
Transformation = grain.Transformation | tr_abc.Transformation


_KD_TO_GRAIN_TRANSFORM = {
    tr_abc.MapTransform: grain.MapTransform,
    # tr_abc.RandomMapTransform: grain.RandomMapTransform,
    tr_abc.FilterTransform: grain.FilterTransform,
}


def apply_transformations(
    ds: tf.data.Dataset,
    transforms: list[Transformation],
) -> tf.data.Dataset:
  """Wrapper around grain to apply the transformations."""
  transforms = [_normalize_transform(tr) for tr in transforms]
  return grain_transforms.apply_transformations(ds, transforms, strict=True)


def _normalize_transform(tr: Transformation) -> grain.Transformation:
  """Convert the kd transform to a `grain.Transformation`."""
  return tr_normalize.normalize_transform(
      tr,
      grain_module=grain,
      kd_to_grain_transform=_KD_TO_GRAIN_TRANSFORM,
      grain_transform_to_apply_wrapper=_GRAIN_TRANSFORM_TO_APPLY_WRAPPER,
  )


def _apply_wrapper(
    cls: type[grain.Transformation],
    *,
    fn_name: str,
    restore_meta_features: bool,
) -> None:
  """Wrap the function to remove the grain internals."""
  old_fn = getattr(cls, fn_name)
  new_fn = wrap_map(old_fn, restore_meta_features=restore_meta_features)
  setattr(cls, fn_name, new_fn)


_GRAIN_TRANSFORM_TO_APPLY_WRAPPER = {
    grain.MapTransform: functools.partial(
        _apply_wrapper, fn_name='map', restore_meta_features=True
    ),
    grain.FilterTransform: functools.partial(
        _apply_wrapper, fn_name='filter', restore_meta_features=False
    ),
}


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
