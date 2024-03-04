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

"""Jax sharding utils."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import functools
import typing
from typing import TypeVar

import jax
from kauldron.typing import PyTree  # pylint: disable=g-importing-member
import numpy as np

if typing.TYPE_CHECKING:
  from kauldron.train import train_step  # pylint: disable=g-bad-import-order

# Sharding value can be:
# * None: Propagate current sharding
# * jax.sharding.Sharding: Use this sharding for all sub-tree
# * Callable: Lazily compute the sharding from the array sub-tree
ShardingTree = PyTree[
    None
    | jax.sharding.Sharding
    | Callable[[PyTree[jax.Array]], 'ShardingTree']
]
_T = TypeVar('_T')


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShardingStrategy:
  """Sharding strategy.

  This class defines the sharding for the dataset, params, optimizer, etc...

  Each sharding can be a PyTree with leafs being one of:

  * None: No sharding specified, `jax.jit` will auto-compute the sharding
  * jax.sharding.Sharding: Use this sharding for all sub-tree
  * Callable: Lazily compute the sharding from the array sub-tree
  """

  ds: ShardingTree = dataclasses.field(
      default_factory=lambda: sharding.FIRST_DIM  # pytype: disable=name-error
  )

  params: ShardingTree = dataclasses.field(
      default_factory=lambda: sharding.REPLICATED  # pytype: disable=name-error
  )
  collections: ShardingTree = None
  # Use `None` to auto-propagate the sharding from model sharding
  opt_state: ShardingTree = None
  # TODO(epot): Should auto-propagate sharding for auxiliaries, but currently
  # image summaries propagate the wrong sharding, like: xid/97663348
  aux: ShardingTree = dataclasses.field(
      default_factory=lambda: sharding.REPLICATED  # pytype: disable=name-error
  )

  @property
  def state(self) -> train_step.TrainState:
    """State sharding."""
    from kauldron.train import train_step  # pylint: disable=g-import-not-at-top

    return train_step.TrainState(  # pytype: disable=wrong-arg-types
        step=sharding.REPLICATED,
        params=self.params,
        collections=self.collections,
        opt_state=self.opt_state,
        # TODO(epot): Remove.
        training_time_hours=sharding.REPLICATED,
    )


# Use class so methods are lazily called. Otherwise, `jax.devices` fail because
# `app.run(main)` is not yet called
class _ShardingAPI:
  """Jax sharding utils.

  Usage:

  ```python
  # Single device array
  params = nn.Sequential(...).init(...)

  # Replicate the array across multiple processes (support multi-host)
  kd.sharding.device_put(array, kd.sharding.REPLICATED)
  ```
  """

  @functools.cached_property
  def _global_mesh(self) -> jax.sharding.Mesh:
    devices = np.asarray(jax.devices())
    return jax.sharding.Mesh(devices, axis_names=('devices',))

  @functools.cached_property
  def _local_mesh(self) -> jax.sharding.Mesh:
    devices = np.asarray(jax.local_devices())
    return jax.sharding.Mesh(devices, axis_names=('devices',))

  @functools.cached_property
  def REPLICATED(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self._global_mesh, jax.sharding.PartitionSpec()
    )

  @functools.cached_property
  def FIRST_DIM(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self._global_mesh, jax.sharding.PartitionSpec('devices')
    )

  @functools.cached_property
  def _LOCAL_REPLICATED(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self._local_mesh, jax.sharding.PartitionSpec()
    )

  @functools.cached_property
  def _LOCAL_FIRST_DIM(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self._local_mesh, jax.sharding.PartitionSpec('devices')
    )

  @functools.cached_property
  def _process_count(self) -> int:
    return jax.process_count()

  def device_put(self, arrays: _T, sharding: jax.sharding.NamedSharding) -> _T:  # pylint: disable=redefined-outer-name
    """Similar to `jax.device_put`, but support multi-process.

    Args:
      arrays: The nested tree of array to shard/replicate
      sharding: How to shard the array. Currently, only `kd.sharding.REPLICATED`
        and `kd.sharding.FIRST_DIM` are supported.

    Returns:
      The replicated nested tree of array
    """
    return jax.tree_map(
        functools.partial(self._put_device_single, sharding=sharding), arrays
    )

  def _put_device_single(
      self,
      array: _T,
      *,
      sharding: jax.sharding.NamedSharding,  # pylint: disable=redefined-outer-name
  ) -> _T:
    """Shard single element."""
    if sharding is self.FIRST_DIM:
      array = jax.device_put(array, self._LOCAL_FIRST_DIM)
      global_shape = (array.shape[0] * self._process_count,) + array.shape[1:]
    elif sharding is self.REPLICATED:
      array = jax.device_put(array, self._LOCAL_REPLICATED)
      global_shape = array.shape
    else:
      raise ValueError(f'Unsupported sharding: {sharding!r}')

    return jax.make_array_from_single_device_arrays(
        global_shape,
        sharding,
        [shard.data for shard in array.addressable_shards],
    )

  def with_sharding_constraint(
      self,
      x: _T,
      shardings: ShardingTree,
  ) -> _T:
    """Wrapper around `jax.lax.with_sharding_constraint`.

    Supports:

    * Optional (`None`) or lazy sharding (`Callable`)
    * Better typing annotations

    Each leaf of the sharding pytree can be:

    * None: Propagate current sharding (auto-inferred by jax)
    * jax.sharding.Sharding: Use this sharding for all sub-tree
    * Callable: Lazily compute the sharding from the array sub-tree

    Args:
      x: PyTree of jax.Arrays which will have their shardings constrained
      shardings: PyTree of sharding specifications. Each leaf can be None,
        sharding or Callable

    Returns:
      The sharded array.
    """

    def _merge(s, x_):
      if s is None:  # No sharding provided, forward the original sharding
        return x_
      elif callable(s):  # Lazy sharding, resolving & recurse
        return self.with_sharding_constraint(x_, s(x_))
      else:
        return jax.lax.with_sharding_constraint(x_, s)

    return jax.tree_map(
        _merge,
        shardings,
        x,
        is_leaf=lambda x: x is None,
    )

  ShardingStrategy = ShardingStrategy  # pylint: disable=invalid-name

  # Typing annotation to annotate sharding pytree
  ShardingTree = ShardingTree  # pylint: disable=invalid-name


sharding = _ShardingAPI()
