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

"""Normalize utils."""

from __future__ import annotations

from typing import Any, Callable, Type

import grain.python as pygrain
# TODO(klausg): split this file into 3 parts to isolate the grain deps
import grain.tensorflow as tfgrain
from kauldron.data.tf import grain_utils
from kauldron.data.transforms import abc as tr_abc


Transformation = (
    tr_abc.Transformation
    | pygrain.Transformation
    | tfgrain.Transformation
    # Treat callables as MapTransforms to keep support for
    # grand-vision preprocessing ops.
    | Callable[[Any], Any]
)


class TransformAdapter:
  """Base class for adapters from Kauldron transforms to grain transforms."""

  def __init__(self, transform):
    self.transform = transform

  def __repr__(self):
    """Return the repr of the wrapped transform with the adaptor as prefix."""
    return f'{self.__class__.__name__}({self.transform!r})'


class TfGrainMapAdapter(TransformAdapter, tfgrain.MapTransform):
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


class TfGrainCallableAdapter(TransformAdapter, tfgrain.MapTransform):
  """Adapter for any callable to a tfgrain MapTransform."""

  def map(self, element: Any) -> Any:
    # Required due to b/326590491.
    meta_features, ex_features = grain_utils.split_grain_meta_features(element)
    out = self.transform(ex_features)
    return grain_utils.merge_grain_meta_features(meta_features, out)


class TfGrainFilterAdapter(TransformAdapter, tfgrain.FilterTransform):
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


class PyGrainMapAdapter(TransformAdapter, pygrain.MapTransform):
  """Adapter from `kd.data.MapTransform` to pygrain."""

  def map(self, element: Any) -> Any:
    return self.transform.map(element)


class PyGrainFilterAdapter(TransformAdapter, pygrain.FilterTransform):
  """Adapter from `kd.data.FilterTransform` to pygrain."""

  def filter(self, element: Any) -> bool:
    return self.transform.filter(element)


class PyGrainCallableAdapter(TransformAdapter, pygrain.MapTransform):
  """Adapter for any callable to a pygrain MapTransform."""

  def map(self, element: Any) -> Any:
    return self.transform(element)


_KD_TO_TFGRAIN_ADAPTERS: dict[Any, Type[tfgrain.Transformation]] = {
    tr_abc.MapTransform: TfGrainMapAdapter,
    tr_abc.FilterTransform: TfGrainFilterAdapter,
    Callable: TfGrainCallableAdapter,  # support grand-vision preprocessing ops
}

_KD_TO_PYGRAIN_ADAPTERS: dict[Any, Type[pygrain.Transformation]] = {
    tr_abc.MapTransform: PyGrainMapAdapter,
    tr_abc.FilterTransform: PyGrainFilterAdapter,
    Callable: PyGrainCallableAdapter,
}


def adapt_for_tfgrain(transform: Transformation) -> tfgrain.Transformation:
  if isinstance(transform, tfgrain.Transformation):
    return transform
  return _adapt(transform, _KD_TO_TFGRAIN_ADAPTERS)


def adapt_for_pygrain(transform: Transformation) -> pygrain.Transformation:
  if isinstance(transform, pygrain.Transformation):
    return transform
  return _adapt(transform, _KD_TO_PYGRAIN_ADAPTERS)


def _adapt(transform, adapter_mapping):
  for kd_cls, Adapter in adapter_mapping.items():  # pylint: disable=invalid-name
    if isinstance(transform, kd_cls):
      break
  else:
    raise TypeError(
        f'Unexpected Kauldron transform: {type(transform).__qualname__}. This'
        ' is likely a Kauldron bug. Please report.'
    )
  return Adapter(transform)  # pytype: disable=missing-parameter,wrong-arg-count,not-instantiable
