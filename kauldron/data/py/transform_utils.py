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

"""Utils for using Kauldron transforms with PyGrain."""

from typing import Any, Callable, Mapping

import grain.python as grain
from kauldron.data.transforms import abc as tr_abc
from kauldron.data.transforms import normalize as tr_normalize

_MISSING: Any = object()


class SliceDataset:
  """Transform which select a subset of the dataset.

  Can be useful to debug or train on a subset of the data.

  ```python
  ds = kd.data.py.Tfds(
      name='mnist',
      split='train',
      transforms=[
          kd.data.py.SliceDataset(10),  # Select ds[:10]
      ],
  )
  ```
  """

  # Internally, this is converted to `ds = ds.slice` in `_apply_transform`

  def __init__(
      self,
      start: int | None = _MISSING,
      stop: int | None = _MISSING,
      step: int | None = _MISSING,
  ):
    # Called as `SliceDataset(stop)`
    if start is not _MISSING and stop is _MISSING and step is _MISSING:
      stop = start
      start = _MISSING

    args = (start, stop, step)
    args = [None if arg is _MISSING else arg for arg in args]
    self.slice = slice(*args)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self.slice})"


class PyGrainMapAdapter(tr_normalize.TransformAdapter, grain.MapTransform):
  """Adapter from `kd.data.MapTransform` to pygrain."""

  def map(self, element: Any) -> Any:
    return self.transform.map(element)


class PyGrainFilterAdapter(
    tr_normalize.TransformAdapter, grain.FilterTransform
):
  """Adapter from `kd.data.FilterTransform` to pygrain."""

  def filter(self, element: Any) -> bool:
    return self.transform.filter(element)


class PyGrainCallableAdapter(tr_normalize.TransformAdapter, grain.MapTransform):
  """Adapter for any callable to a pygrain MapTransform."""

  def map(self, element: Any) -> Any:
    return self.transform(element)


_KD_TO_PYGRAIN_ADAPTERS = {
    tr_abc.MapTransform: PyGrainMapAdapter,
    tr_abc.FilterTransform: PyGrainFilterAdapter,
    Callable: PyGrainCallableAdapter,
}


def _adapt_for_pygrain(
    transform: tr_normalize.Transformation,
) -> grain.Transformation:
  if isinstance(transform, (grain.Transformation, SliceDataset)):
    return transform
  return tr_normalize.adapt_transform(transform, _KD_TO_PYGRAIN_ADAPTERS)


def apply_transforms(
    ds: grain.MapDataset, transforms: tr_normalize.Transformations
) -> grain.MapDataset:
  """Apply the transformations to the dataset."""
  if isinstance(transforms, Mapping):
    transforms = transforms.values()
  for tr in transforms:
    tr = _adapt_for_pygrain(tr)
    ds = _apply_transform(ds, tr)
  return ds


def _apply_transform(
    ds: grain.MapDataset, tr: grain.Transformation
) -> grain.MapDataset:
  """Apply a list of single transformation."""
  match tr:
    case grain.MapTransform():
      ds = ds.map(tr)
    case grain.RandomMapTransform():
      ds = ds.random_map(tr)
    case grain.FilterTransform():
      ds = ds.filter(tr)
    case grain.Batch():
      ds = ds.batch(tr.batch_size, drop_remainder=tr.drop_remainder)
    case SliceDataset():
      ds = ds.slice(tr.slice)
    case _:
      raise ValueError(f"Unexpected transform type: {tr}")
  return ds
