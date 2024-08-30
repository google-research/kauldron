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

"""Small wrapper around `jax.random`."""

from __future__ import annotations

from collections.abc import Iterator
import functools
import hashlib
import sys
import typing
from typing import Any, TypeVar

from etils.epy import _internal
import jax
import jax.random
import numpy as np


_FnT = TypeVar('_FnT')

if typing.TYPE_CHECKING:
  # For type checking, `PRNGKey` is a `jax.Array`
  _Base = jax.Array
else:
  _Base = object


@jax.tree_util.register_pytree_node_class
class PRNGKey(_Base):
  """Small wrapper around `jax.random` key arrays to reduce boilerplate.

  Benefits:

  * Object oriented API (`jax.random.uniform(key)` -> `key.uniform()`)
  * `fold_in` supports `str` (`key.fold_in('dropout')`)
  * Additional `as_seed()` method to get a seed `int` from the rng (to pass
    to third party APIs, like `grain`, `np.random`,...)

  Usage:

  ```python
  key = kd.random.PRNGKey(0)
  key0, key1 = key.split()
  x = key0.uniform()

  x = jax.random.uniform(key)  # Jax API still works
  ```
  """

  rng: jax.Array

  def __init__(
      self,
      seed_or_rng: int | jax.Array | PRNGKey = 0,
      *,
      impl: str | None = None,
      _allow_seed_array: bool = True,  # Internal variable
  ):
    """Constructor."""

    # TODO(epot): Check that key is only used once ? (except on Colab)
    # Fold-in should also not be called on the same value twice.

    if isinstance(seed_or_rng, PRNGKey):
      self.rng = seed_or_rng.rng  # type: ignore[annotation-type-mismatch]
    elif (
        _allow_seed_array
        and isinstance(seed_or_rng, jax.Array)
        and not jax.dtypes.issubdtype(seed_or_rng.dtype, jax.dtypes.prng_key)
        and seed_or_rng.shape == ()  # pylint: disable=g-explicit-bool-comparison
    ):
      # `kd_random.PRNGKey(jnp.asarray(0))` is a valid seed.
      self.rng = jax.random.PRNGKey(seed_or_rng, impl=impl)
    elif isinstance(
        seed_or_rng,
        (
            jax.Array,
            # `jax.core.ShapedArray` is created by `flax.linen.scan`
            jax.core.ShapedArray,
        ),
    ):
      self.rng = seed_or_rng
    elif hasattr(jax.random, 'PRNGKeyArray') and isinstance(
        seed_or_rng, jax.random.PRNGKeyArray
    ):
      # In jax versions before 0.4.16, typed PRNG keys have a special type.
      # In newer versions, typed PRNG keys pass the Array instance check above.
      self.rng = seed_or_rng  # type: ignore[assignment]
    elif type(seed_or_rng) is object:  # pylint: disable=unidiomatic-typecheck
      # Checking for `object` is a hack required for `@jax.vmap` compatibility:
      # In `jax/_src/api_util.py` for `flatten_axes`, jax set all values to a
      # dummy sentinel `object()` value.
      self.rng = seed_or_rng  # type: ignore[assignment]
    else:  # `int` or `np.ndarray`, normalize
      self.rng = jax.random.PRNGKey(seed_or_rng, impl=impl)

  def __iter__(self) -> Iterator[PRNGKey]:
    return (self._new(k) for k in iter(self.rng))

  def __getitem__(self, slice_) -> PRNGKey:
    return self._new(self.rng[slice_])

  def split(self, n: int = 2) -> PRNGKey:
    """Returns the next rng key."""
    return self._new(jax.random.split(self, n))

  def fold_in(self, data: int | str) -> PRNGKey:
    """Folds in delta into the random state."""
    if isinstance(data, str):
      data = _hash(data)
    return self._new(jax.random.fold_in(self, data))

  def next(self) -> PRNGKey:
    """Returns the next rng key (alias for `key.split(1)[0]`)."""
    return self.split(1)[0]

  def _new(self, key) -> PRNGKey:
    return type(self)(key, _allow_seed_array=False)

  def __repr__(self):
    return f'{type(self).__name__}({self.rng!r})'

  def tree_flatten(self) -> tuple[list[jax.Array], dict[str, Any]]:
    """`jax.tree_utils` support."""
    return ([self.rng], {})  # type: ignore[bad-return-type]

  @classmethod
  def tree_unflatten(
      cls,
      metadata: dict[str, Any],
      array_field_values: list[jax.Array],
  ) -> PRNGKey:
    """`jax.tree_utils` support."""
    del metadata
    (array_field_values,) = array_field_values
    # Support tree_map when the output is None or array normalization
    # e.g.
    # * `chex.assert_trees_all_close` normalize to `np.ndarray`
    # * `jax.tree.map(np.testing.assert_allclose)`
    if array_field_values is None:
      return None
    elif isinstance(array_field_values, np.ndarray):
      return array_field_values
    else:
      rng = cls(array_field_values)
      return rng

  def __array__(self, dtype=None, copy=None) -> np.ndarray:
    """Support np.array conversion `np.asarray(key)`."""
    assert dtype is None
    assert copy is None
    return np.asarray(self.rng)

  def as_seed(self) -> int:
    """Returns a `seed` integer (alias of `int(rng.bits())`).

    Note this is non-reversible (the returned seed is not the one passed to
    construct the rng).

    Returns:
      An integer seed.
    """
    return int(self.bits())  # pytype: disable=missing-parameter

  ball = jax.random.ball
  bernoulli = jax.random.bernoulli
  beta = jax.random.beta
  bits = jax.random.bits
  categorical = jax.random.categorical
  cauchy = jax.random.cauchy
  chisquare = jax.random.chisquare
  choice = jax.random.choice
  dirichlet = jax.random.dirichlet
  double_sided_maxwell = jax.random.double_sided_maxwell
  exponential = jax.random.exponential
  f = jax.random.f
  gamma = jax.random.gamma
  generalized_normal = jax.random.generalized_normal
  geometric = jax.random.geometric
  gumbel = jax.random.gumbel
  key_data = jax.random.key_data
  laplace = jax.random.laplace
  logistic = jax.random.logistic
  loggamma = jax.random.loggamma
  laplace = jax.random.laplace
  logistic = jax.random.logistic
  maxwell = jax.random.maxwell
  multivariate_normal = jax.random.multivariate_normal
  normal = jax.random.normal
  orthogonal = jax.random.orthogonal
  pareto = jax.random.pareto
  permutation = jax.random.permutation
  poisson = jax.random.poisson
  rademacher = jax.random.rademacher
  randint = jax.random.randint
  rayleigh = jax.random.rayleigh
  t = jax.random.t
  truncated_normal = jax.random.truncated_normal
  uniform = jax.random.uniform
  wald = jax.random.wald
  weibull_min = jax.random.weibull_min


@functools.lru_cache(maxsize=1_000)
def _hash(data: str) -> int:
  """Deterministic hash."""
  data = hashlib.sha1(data.encode('utf-8')).digest()
  data = int.from_bytes(data[:4], byteorder='big')  # Truncate to uint32
  return data


@functools.cache
def _mock_jax():
  """Mock `jax.random` to support custom Key object."""
  from jax._src import random  # pylint: disable=g-import-not-at-top
  from flax.core import scope  # pylint: disable=g-import-not-at-top

  random._check_prng_key = _normalize_jax_key(  # pylint: disable=protected-access
      random._check_prng_key, key_arg_index=1  # pylint: disable=protected-access
  )
  scope._is_valid_rng = _normalize_jax_key(scope._is_valid_rng)  # pylint: disable=protected-access


def _normalize_jax_key(fn: _FnT, *, key_arg_index: int = 0) -> _FnT:
  """Mock `jax.random` to support custom Key object."""

  fn = _internal.unwrap_on_reload(fn)  # pylint: disable=protected-access

  # Support Colab reload

  @_internal.wraps_with_reload(fn)
  def new_fn(*args, **kwargs):
    nonlocal key_arg_index
    if len(args) == 1:  # Colab backward compatibility (old kernels)
      key_arg_index = 0  # TODO(epot): Remove once all kernels are updated

    key = args[key_arg_index]
    scope = sys.modules.get('flax.core.scope')
    if isinstance(key, PRNGKey):
      key = key.rng
    elif scope is not None and isinstance(key, scope.LazyRng):
      key = key.as_jax_rng()
    new_args = list(args)
    new_args[key_arg_index] = key
    return fn(*new_args, **kwargs)

  return new_fn


# We mock jax for compatibility
_mock_jax()
