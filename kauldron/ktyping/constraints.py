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

"""Constraints is an immutable mapping of dimension names to Constraint."""

from __future__ import annotations

from collections.abc import Mapping
import typing
from typing import Any, Iterable, Iterator, Optional

from kauldron.ktyping.internal_typing import Undefined  # pylint: disable=g-importing-member

# TODO(klausg): not too happy about these names.
# maybe Assignments? DimValues? ShapeInfo?
# also how to call Alternatives? The concept is a bit unintuitive, and the name
# does not help much.
Constraint: typing.TypeAlias = tuple[int | Undefined, ...]
ConstraintAlternatives: typing.TypeAlias = frozenset["Constraints"]


class Constraints(Mapping[str, Constraint]):
  """Store the mappings of dim names to known values.

   An immutable wrapper around dictionaries that implements the complete
  `collections.Mapping`  interface.
  """

  @classmethod
  def fromkeys(
      cls, seq: Iterable[str], value: Optional[Constraint] = None
  ) -> Constraints:
    return cls(dict.fromkeys(seq, value))

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    self._dict = dict(*args, **kwargs)
    self._hash: Optional[int] = None

  def __getitem__(self, key: str) -> Constraint:
    return self._dict[key]

  def __contains__(self, key: object) -> bool:
    return key in self._dict

  def copy(self, **add_or_replace: Constraint) -> Constraints:
    return self.__class__(self, **add_or_replace)

  def delete(self, key: str) -> Constraints:
    new = dict(self._dict)
    del new[key]
    return self.__class__(new)

  def __iter__(self) -> Iterator[str]:
    return iter(self._dict)

  def __len__(self) -> int:
    return len(self._dict)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self._dict!r})"

  def __hash__(self) -> int:
    if self._hash is None:
      h = 0
      for key, value in self.items():
        h ^= hash((key, value))
      self._hash = h

    return self._hash

  def __or__(self, other: Any) -> Constraints:
    if not isinstance(other, (dict, self.__class__)):
      return NotImplemented  # pytype: disable=bad-return-type
    new = dict(self)
    new.update(other)
    return self.__class__(new)

  def __ror__(self, other: Any) -> dict[Any, Any]:
    if not isinstance(other, (dict, self.__class__)):
      return NotImplemented  # pytype: disable=bad-return-type
    new = dict(other)
    new.update(self)
    return new

  def __ior__(self, other: Any) -> None:
    raise TypeError(f"{self.__class__.__name__} object is not mutable")
