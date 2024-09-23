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

from typing import Any, Callable, Sequence, TypeVar

import grain.python as pygrain
from kauldron.data.transforms import abc as tr_abc


Transformation = (
    tr_abc.Transformation
    # using pygrain.Transformation here is sufficient because tfgrain transforms
    # inherit from pygrain.Transformation.
    | pygrain.Transformation
    # Treat callables as MapTransforms to keep support for
    # grand-vision preprocessing ops.
    | Callable[[Any], Any]
)
Transformations = Sequence[Transformation] | dict[str, Transformation]


class TransformAdapter:
  """Base class for adapters from Kauldron transforms to grain transforms."""

  def __init__(self, transform):
    self.transform = transform

  def __repr__(self):
    """Return the repr of the wrapped transform with the adaptor as prefix."""
    return f'{self.__class__.__name__}({self.transform!r})'


T = TypeVar('T', bound=TransformAdapter)


def adapt_transform(
    transform: Transformation,
    adapter_mapping: dict[Any, type[T]],
) -> T:
  """Wrap the transform in the appropriate adapter as per adapter_mapping."""
  for kd_cls, Adapter in adapter_mapping.items():  # pylint: disable=invalid-name
    if isinstance(transform, kd_cls):
      break
  else:
    raise TypeError(
        f'Unexpected Kauldron transform: {type(transform).__qualname__}. This'
        ' is likely a Kauldron bug. Please report.'
    )
  return Adapter(transform)  # pytype: disable=missing-parameter,wrong-arg-count,not-instantiable
