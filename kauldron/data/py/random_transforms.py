# Copyright 2025 The kauldron Authors.
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

"""pygrain random map transforms."""

import abc
import copy
import dataclasses
from typing import Any
from typing import Optional
from grain import python as pygrain
from kauldron.data.transforms import base
from kauldron.typing import Array, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseRandomTransform(base.ElementWise, pygrain.RandomMapTransform):  # pylint: disable=protected-access
  """Base class for elementwise transforms."""

  def random_map(self, features, rng: np.random.Generator):
    features_out = {}
    # All sub-keys will share the same sub_rng random state, which we spawn
    # ONCE from the main rng (to correctly advance the main rng).
    # This is useful e.g. when applying the same random cropping to multiple
    # modalities like rgb and depth.
    (sub_rng,) = rng.spawn(1)
    for key, element, should_transform in self._per_element(features):
      if should_transform:
        elem_rng = copy.deepcopy(sub_rng)  # copy the SAME rng for each element.
        features_out[key] = self.random_map_element(element, elem_rng)
      else:
        features_out[key] = element
    return features_out

  @abc.abstractmethod
  def random_map_element(self, element: Any, rng: np.random.Generator):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True)
class RandomFlipLeftRight(ElementWiseRandomTransform):
  """Flips an image horizontally with probability 50%."""

  @typechecked
  def random_map_element(
      self, element: Array["... H W C"], rng: np.random.Generator
  ) -> Array["... H W C"]:
    bcastable_batch_shape = element.shape[:-3] + (1, 1, 1)
    mask = rng.integers(2, size=bcastable_batch_shape, dtype=bool)
    return np.where(mask, element, np.flip(element, axis=-2))


@dataclasses.dataclass(kw_only=True, frozen=True)
class RandomCrop(ElementWiseRandomTransform):
  """Randomly crop the input data to the specified shape.

  Can be used on data of any shape or type including images and videos.

  Attributes:
    shape: A tuple of integers describing the target shape of the crop. Entries
      can be also be None to keep the original shape of the data in that dim.
  """

  shape: tuple[Optional[int], ...]

  def __post_init__(self):
    if not all([d is None or d >= 0 for d in self.shape]):
      raise ValueError(
          "Target shape can contain only non-negative ints or None. Got"
          f" {self.shape=}"
      )
    super().__post_init__()

  def random_map(self, features, rng: np.random.Generator):
    shapes = {k: v.shape for k, v in features.items() if k in self.key}
    for key, shape in shapes.items():
      if len(shape) != len(self.shape):
        raise ValueError(
            "Rank of self.shape has to match element shape. But got"
            f" {self.shape=} and {shape=} for {key!r}"
        )
    ref_key, ref_shape = next(iter(shapes.items())) if shapes else (None, None)
    # ensure dimensions match except where self.shape is None
    for key, shape in shapes.items():
      for ref_dim, key_dim, target_dim in zip(ref_shape, shape, self.shape):
        if ref_dim != key_dim and (target_dim is not None):
          raise ValueError(
              "Shapes of different keys for random crop have to be compatible,"
              f" but got {ref_shape} ({ref_key}) != {shape} ({key}) with"
              f" {self.shape=}"
          )

    return super().random_map(features, rng)

  @typechecked
  def random_map_element(
      self, element: Array["..."], rng: np.random.Generator
  ) -> Array["..."]:
    # Dimensions for which to keep the original size
    effective_shape = tuple(
        element.shape[i] if s is None else s for i, s in enumerate(self.shape)
    )

    if any(e < s for e, s in zip(element.shape, effective_shape)):
      raise ValueError(
          f"Element shape {element.shape} is smaller than the crop shape"
          f" {self.shape}."
      )

    slices = []
    for i, crop_size in enumerate(effective_shape):
      dim_size = element.shape[i]
      if crop_size < dim_size:
        start = rng.integers(0, dim_size - crop_size + 1)
        slices.append(slice(start, start + crop_size))
      else:
        slices.append(slice(None))

    return element[tuple(slices)]
