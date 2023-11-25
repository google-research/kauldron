# Copyright 2023 The kauldron Authors.
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

"""Immutable dict util."""

from __future__ import annotations

from collections.abc import Hashable
import sys
from typing import Any, ClassVar

from etils import epy
import immutabledict as immutabledict_lib


class ImmutableDict(immutabledict_lib.immutabledict):
  """Immutable dict abstraction with `getattr` access."""

  _dca_jax_tree_registered: ClassVar[bool] = False

  def __new__(cls, *args: Any, **kwargs: Any) -> ImmutableDict:
    del args, kwargs  # Unused
    if not cls._dca_jax_tree_registered and 'jax' in sys.modules:
      import jax  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

      jax.tree_util.register_pytree_with_keys_class(cls)
      cls._dca_jax_tree_registered = True
    return super().__new__(cls)  # pylint: disable=no-value-for-parameter

  def __getattr__(self, name: str) -> str:
    # The base-class has a `dict_cls` attribute, but collisions should be
    # extremely rare.
    return self[name]

  def __repr__(self) -> str:
    return epy.Lines.make_block(
        header=f'{self.__class__.__name__}',
        content={repr(k): v for k, v in self._dict.items()},
        braces=('({', '})'),
        equal=': ',
    )

  def tree_flatten_with_keys(self) -> tuple[tuple[Any, ...], Hashable]:
    """Flattens this FrozenDict.

    Returns:
      A flattened version of this FrozenDict instance.
    """
    import jax  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    sorted_keys = sorted(self)
    return tuple(
        [(jax.tree_util.DictKey(k), self[k]) for k in sorted_keys]
    ), tuple(self)

  @classmethod
  def tree_unflatten(cls, keys, values):
    # Flatten sort the keys, so reconstruct the ordered sorted
    ordered_items = {k: v for k, v in zip(sorted(keys), values)}
    # Restore original dict order
    new_items = ((k, ordered_items[k]) for k in keys)

    return cls(new_items)
