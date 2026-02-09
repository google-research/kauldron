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

"""PyGrain Preprocessing ops."""

from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Sequence

import grain.python as grain
from jax import tree_util
from kauldron.typing import Array, PyTree, typechecked  # pylint: disable=g-importing-member,g-multiple-import
import numpy as np


# Alias of `third_party/py/jax/_src/tree_util.py:KeyPath` which is not included
# in public API.
KeyEntry = Any
KeyPath = tuple[KeyEntry, ...]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseTransformWithPredicate(grain.MapTransform):
  """Base class for elementwise transforms.

  This class is intended to allow key and/or feature-dependent logic to
  determine whether to apply a particular transformation, e.g. running a
  rescaling operation on keys containing 'rgb' or 'image'.
  """

  # A predicate accepting key (or KeyPath, if `features` is a PyTree) and
  # value inputs and returning whether to transform that feature.
  should_transform_pred: Callable[[str | KeyPath, Any], bool] = (
      lambda k, v: False
  )

  def map(self, features: PyTree[Any]) -> PyTree[Any]:
    def maybe_transform(path: KeyPath, feature: Any):
      should_transform = self.should_transform_pred(path, feature)
      return self.map_element(feature) if should_transform else feature

    return tree_util.tree_map_with_path(maybe_transform, features)

  @abc.abstractmethod
  def map_element(self, element):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseRandomTransformWithPredicate(grain.RandomMapTransform):
  """Base class for elementwise random transforms.

  This class is intended to allow key and/or feature-dependent logic to
  determine whether to apply a particular transformation, e.g. running a
  rescaling operation on keys containing 'rgb' or 'image'.
  """

  # A predicate accepting key (or KeyPath, if `features` is a PyTree) and
  # value inputs and returning whether to transform that feature.
  should_transform_pred: Callable[[str | KeyPath, Any], bool] = (
      lambda k, v: False
  )

  def random_map(
      self, features: PyTree[Any], rng: np.random.Generator
  ) -> PyTree[Any]:
    def maybe_transform(path: KeyPath, feature: Any):
      should_transform = self.should_transform_pred(path, feature)
      return (
          self.random_map_element(feature, rng) if should_transform else feature
      )

    return tree_util.tree_map_with_path(maybe_transform, features)

  @abc.abstractmethod
  def random_map_element(self, element, rng):
    raise NotImplementedError


class FlipUpsideDown(ElementWiseTransformWithPredicate):
  """Flips an image vertically (upside down)."""

  @typechecked
  def map_element(self, element: Array["*B H W C"]) -> Array["*B H W C"]:
    return np.flip(element, axis=-3)

  @classmethod
  def matching_keys(
      cls,
      strs: str | Sequence[str] = ("img", "image", "rgb", "cam"),
      strs_to_exclude: str | Sequence[str] = (),
  ):
    """Builds a `FlipUpsideDown` matching specific strings against the keys."""
    strs = [strs] if isinstance(strs, str) else strs
    strs_to_exclude = (
        [strs_to_exclude]
        if isinstance(strs_to_exclude, str)
        else strs_to_exclude
    )

    def should_transform_pred(k: str | KeyPath, _):
      if isinstance(k, tuple) and isinstance(k[-1], tree_util.DictKey):
        leaf_key = k[-1].key
      elif isinstance(k, str):
        leaf_key = k
      else:
        raise ValueError(f"Unsupported key type: {type(k)}")

      matched = any([s in leaf_key for s in strs])
      to_exclude = any([s in leaf_key for s in strs_to_exclude])
      return matched and (not to_exclude)

    return cls(should_transform_pred=should_transform_pred)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PadImage(ElementWiseTransformWithPredicate):
  """Pad image."""

  pad: int
  mode: str = "constant"

  @typechecked
  def map_element(self, element: Array["*b H W C"]) -> Array["*b H2 W2 C"]:
    batch_dims = len(element.shape[:-3])
    padding = ((0, 0),) * batch_dims + (
        (self.pad, self.pad),
        (self.pad, self.pad),
        (0, 0),
    )
    return np.pad(element, padding, mode=self.mode)

  @classmethod
  def matching_keys(
      cls,
      pad: int,
      mode: str = "constant",
      strs: str | Sequence[str] = ("img", "image", "rgb", "cam"),
      strs_to_exclude: str | Sequence[str] = (),
  ):
    """Builds a `PadImage` matching specific strings against the keys."""
    strs = [strs] if isinstance(strs, str) else strs
    strs_to_exclude = (
        [strs_to_exclude]
        if isinstance(strs_to_exclude, str)
        else strs_to_exclude
    )

    def should_transform_pred(k: str | KeyPath, _):
      if isinstance(k, tuple) and isinstance(k[-1], tree_util.DictKey):
        leaf_key = k[-1].key
      elif isinstance(k, str):
        leaf_key = k
      else:
        raise ValueError(f"Unsupported key type: {type(k)}")

      matched = any([s in leaf_key for s in strs])
      to_exclude = any([s in leaf_key for s in strs_to_exclude])
      return matched and (not to_exclude)

    return cls(pad=pad, mode=mode, should_transform_pred=should_transform_pred)
