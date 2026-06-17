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

"""Small wrapper around `jax.random`."""

from __future__ import annotations

from collections.abc import Iterator
import functools
import hashlib
from typing import Any

import jax
import jax.random
import numpy as np


@functools.lru_cache(maxsize=1_000)
def _hash_str_to_int(s: str) -> int:
  """Deterministically hash a string to a 32-bit integer."""
  data = hashlib.sha1(s.encode('utf-8')).digest()
  data = int.from_bytes(data[:4], byteorder='big')  # Truncate to uint32
  return data


def fold_in_str(key: jax.Array, data: str) -> jax.Array:
  """Fold a string into a JAX random key."""
  return jax.random.fold_in(key, _hash_str_to_int(data))


def random_seed(key: jax.Array) -> int:
  """Extract a 32-bit integer seed from a JAX random key."""
  return int(jax.random.bits(key)) % (2**32)


@jax.tree_util.register_pytree_node_class
class PRNGKey:
  """Deprecated compatibility wrapper around jax.random.PRNGKey."""

  rng: jax.Array

  def __init__(self, seed_or_rng: Any = 0, *, impl: str | None = None):
    if isinstance(seed_or_rng, PRNGKey):
      self.rng = seed_or_rng.rng
    elif isinstance(seed_or_rng, int):
      self.rng = jax.random.PRNGKey(seed_or_rng, impl=impl)
    elif (
        isinstance(seed_or_rng, jax.Array)
        and seed_or_rng.ndim == 0
        and not jax.dtypes.issubdtype(seed_or_rng.dtype, jax.dtypes.prng_key)
    ):
      self.rng = jax.random.PRNGKey(seed_or_rng, impl=impl)
    else:
      self.rng = seed_or_rng

  def split(self, n: int = 2) -> PRNGKey:
    return PRNGKey(jax.random.split(self.rng, n))

  def fold_in(self, data: int | str) -> PRNGKey:
    if isinstance(data, str):
      data = _hash_str_to_int(data)
    return PRNGKey(jax.random.fold_in(self.rng, data))

  def as_seed(self) -> int:
    return random_seed(self.rng)

  def next(self) -> PRNGKey:
    return self.split(1)[0]

  def __iter__(self) -> Iterator[PRNGKey]:
    return (PRNGKey(k) for k in iter(self.rng))

  def __getitem__(self, slice_) -> PRNGKey:
    return PRNGKey(self.rng[slice_])

  def __len__(self) -> int:
    return len(self.rng)

  def tree_flatten(self) -> tuple[list[jax.Array], dict[str, Any]]:
    return ([self.rng], {})

  @classmethod
  def tree_unflatten(
      cls, metadata: dict[str, Any], array_field_values: list[jax.Array]
  ) -> PRNGKey:
    del metadata
    (array_field_values,) = array_field_values
    if array_field_values is None:
      return None  # type: ignore
    elif isinstance(array_field_values, np.ndarray):
      return array_field_values  # type: ignore
    else:
      return cls(array_field_values)

  def __array__(self, dtype=None, copy=None) -> np.ndarray:
    assert dtype is None
    assert copy is None
    return np.asarray(self.rng)

  def __repr__(self):
    return f'{type(self).__name__}({self.rng!r})'

  def __getattr__(self, name: str) -> Any:
    if hasattr(jax.random, name):
      method = getattr(jax.random, name)
      if callable(method):

        def wrapper(*args, **kwargs):
          return method(self.rng, *args, **kwargs)

        return wrapper
    return getattr(self.rng, name)
