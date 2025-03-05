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

"""Base transform class."""

import abc
from collections.abc import Iterable, Mapping
import dataclasses
import typing
from typing import Any, Sequence

import flax
from kauldron import kontext
from kauldron.data.transforms import abc as tr_abc

_FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Elements(tr_abc.MapTransform):
  """Modify the elements by keeping xor dropping and/or renaming and/or copying."""

  keep: Iterable[str] = ()
  drop: Iterable[str] = ()
  rename: Mapping[str, str] = _FrozenDict()
  copy: Mapping[str, str] = _FrozenDict()

  def __post_init__(self):
    if self.keep and self.drop:
      raise ValueError("keep and drop are mutually exclusive")
    keep = set(self.keep)
    drop = set(self.drop)

    rename_keys = set(self.rename.keys())
    rename_values = set(self.rename.values())

    keep_and_rename = keep & (rename_keys | rename_values)
    if keep_and_rename:
      raise KeyError(
          f"Keys: {keep_and_rename} present in both keep and "
          "rename (key or value) collections."
      )

    drop_and_rename = drop & rename_keys
    if drop_and_rename:
      raise KeyError(
          f"Keys: {drop_and_rename} present in both drop and "
          "rename (key) collections."
      )

    copy_values = set(self.copy.values())
    if len(copy_values) != len(self.copy.values()):
      raise ValueError("Copy values must be unique.")

    overlap_keys = copy_values & (keep | drop | rename_keys | rename_values)
    if overlap_keys:
      raise KeyError(
          f"Keys: {overlap_keys} present in both copy and "
          "keep or drop or rename (key or value) collections."
      )

    object.__setattr__(self, "keep", keep)
    object.__setattr__(self, "drop", drop)

  def map(self, features):
    feature_keys = set(features.keys())

    # first handle copying because some elements might be dropped/renamed
    copy_output = {}
    # Auto-graph fail to convert the condition, so explicitly set
    # `bool(self.copy)` (see b/335839876)
    if bool(self.copy):
      copy_keys = set(self.copy.keys())
      missing_copy_keys = copy_keys - feature_keys
      if missing_copy_keys:
        raise KeyError(
            f"copy-key(s) {missing_copy_keys} not found in batch. "
            f"Available keys are {sorted(feature_keys)!r}."
        )
      copy_values = set(self.copy.values())
      overlap_keys = copy_values & feature_keys
      if overlap_keys:
        raise KeyError(
            f"copy-value(s) {overlap_keys} will overwrite existing values in "
            f"batch. Existing keys are {sorted(feature_keys)!r}."
        )
      copy_output = {v: features[k] for k, v in self.copy.items()}

    # resolve keep or drop
    if self.keep:
      keep_keys = set(self.keep)
      missing_keep_keys = keep_keys - feature_keys
      if missing_keep_keys:
        raise KeyError(
            f"keep-key(s) {missing_keep_keys} not found in batch. "
            f"Available keys are {sorted(feature_keys)!r}."
        )
      output = {k: v for k, v in features.items() if k in self.keep}
    elif self.drop:
      drop_keys = set(self.drop)
      missing_drop_keys = drop_keys - feature_keys
      if missing_drop_keys:
        raise KeyError(
            f"drop-key(s) {missing_drop_keys} not found in batch. "
            f"Available keys are {sorted(feature_keys)!r}."
        )
      output = {
          k: v
          for k, v in features.items()
          if k not in self.drop and k not in self.rename
      }
    else:  # only rename
      output = {k: v for k, v in features.items() if k not in self.rename}
    output.update(copy_output)

    # resolve renaming
    rename_keys = set(self.rename.keys())
    missing_rename_keys = rename_keys - feature_keys
    if missing_rename_keys:
      raise KeyError(
          f"rename-key(s) {missing_rename_keys} not found in batch. "
          f"Available keys are {sorted(feature_keys)!r}."
      )
    renamed = {
        self.rename[k]: v for k, v in features.items() if k in self.rename
    }
    overwrites = sorted(set(renamed.keys()) & set(output.keys()))
    if overwrites:
      offending_renames = [k for k, v in self.rename.items() if v in overwrites]
      raise KeyError(
          f"Tried renaming key(s) {offending_renames!r} to {overwrites!r} but"
          " target names already exist. Implicit overwriting is not supported."
          " Please explicitly drop target keys that should be overwritten."
      )
    output.update(renamed)
    return output


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class _ElementWise:
  """Mixin class that enables defining a key and iterating relevant elements.

  Mostly used to remove code duplication between ElementWiseTransform and
  ElementWiseRandomTransform.
  """

  key: kontext.Key | Sequence[kontext.Key] | dict[kontext.Key, kontext.Key]

  def __post_init__(self):
    if not self.key:
      raise KeyError(f"kontext.Key required for {self}")

    # Convert key to Dict[kontext.Key, kontext.Key] format.
    keys = self.key
    if isinstance(self.key, str):
      keys = {self.key: self.key}
    elif isinstance(self.key, Sequence):
      keys = {k: k for k in self.key}
    object.__setattr__(self, "key", keys)
    # Is there a more generic way to check that subclasses are calling
    # `super().__post_init__()`?
    object.__setattr__(self, "_post_init_called", True)

  def _per_element(
      self,
      features: dict[str, Any],
  ) -> Iterable[tuple[str, Any, bool]]:
    """Iterator that returns out_key, value, should_transform triplets.

    Args:
      features: a dict of features

    Yields:
      an iterator with tuples of out_key, value, and should_transform.
      `ElementWiseTransform` and `ElementWiseRandomTransform` iterate over this
      and set:
      `batch[out_key] = transformed(value) if should transform else value`

    Raises:
      KeyError: if the key did not match any of the features.
    """
    if not hasattr(self, "_post_init_called"):
      raise ValueError(
          f"{type(self).__qualname__} was not initialized properly. Make sure"
          " to call `super().__post_init__()` inside the sub-class"
          " `def __post_init__`."
      )

    is_noop = True
    for k, v in features.items():
      if k in self.key:
        yield self.key[k], v, True
        if k != self.key[k]:  # if renaming then keep original key
          yield k, v, False
        is_noop = False
      else:
        yield k, v, False
    if is_noop:
      raise KeyError(
          f"{sorted(self.key.keys())} "  # pytype: disable=attribute-error
          "did not match any keys. Available keys: "
          f"{sorted(features.keys())}"  # pytype: disable=attribute-error
      )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseTransform(_ElementWise, tr_abc.MapTransform):
  """Base class for elementwise transforms."""

  def map(self, features):
    return {
        key: self.map_element(element) if should_transform else element
        for key, element, should_transform in self._per_element(features)
    }

  @abc.abstractmethod
  def map_element(self, element):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElementWiseRandomTransformBase(_ElementWise, abc.ABC):
  """Base class for random elementwise transforms."""

  def random_map(self, features, seed):
    features_out = {}
    for key, element, should_transform in self._per_element(features):
      if should_transform:
        features_out[key] = self.random_map_element(element, seed)
      else:
        features_out[key] = element
    return features_out

  @abc.abstractmethod
  def random_map_element(self, element, seed):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TreeFlattenWithPath(_ElementWise, tr_abc.MapTransform):
  """Flatten any tree-structured elements.

  For example, using 'a' as key, with:
    features = {'a': {'b': 2, 'c': {'d': 3}}, 'e': 5 , 'f': {'g': 6}}

  becomes:
    features = {'a_b': 2, 'a_c_d': 3, 'e': 5, 'f': {'g': 6}}
  """

  separator: str = "_"

  def map(self, features):
    output = {}
    for key, element, should_transform in self._per_element(features):
      if should_transform:
        output.update(
            kontext.flatten_with_path({key: element}, separator=self.separator)
        )
      else:
        output[key] = element
    return output


@dataclasses.dataclass(frozen=True, eq=True)
class AddConstants(tr_abc.MapTransform):
  """Adds constant elements.

  ```python
  kd.data.AddConstants({
      'my_field': 1.0,
  })
  ```

  Can be used with mixtures when some datasets have missing fields.
  """

  values: Mapping[str, Any] = flax.core.FrozenDict()

  def map(self, features):
    overwrites = set(self.values.keys()) & set(features.keys())
    if overwrites:
      raise KeyError(
          f"Tried adding key(s) {sorted(overwrites)!r} but"
          " target names already exist. Implicit overwriting is not supported."
          " Please explicitly drop target keys that should be overwritten."
      )
    features.update(self.values)
    return features
