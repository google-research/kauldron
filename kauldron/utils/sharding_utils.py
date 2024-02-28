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

import dataclasses
import functools
import typing
from typing import Any, Optional, TypeVar

import jax
import numpy as np

if typing.TYPE_CHECKING:
  from kauldron.train import train_step  # pylint: disable=g-bad-import-order

_ShardingValue = Any
_T = TypeVar('_T')


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sharding:
  """Sharding informations."""

  ds: _ShardingValue = dataclasses.field(
      default_factory=lambda: sharding.SHARDED  # pytype: disable=name-error
  )

  params: _ShardingValue = dataclasses.field(
      default_factory=lambda: sharding.REPLICATED  # pytype: disable=name-error
  )
  collections: Optional[_ShardingValue] = None
  # Use `None` to auto-propagate the sharding from model sharding
  optimizer: Optional[_ShardingValue] = None
  # TODO(epot): Should auto-propagate sharding for auxiliaries, but currently
  # image summaries propagate the wrong sharding, like: xid/97663348
  aux: Optional[_ShardingValue] = dataclasses.field(
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
        opt_state=self.optimizer,
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
  def SHARDED(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self._global_mesh, jax.sharding.PartitionSpec('devices')
    )

  @functools.cached_property
  def _LOCAL_REPLICATED(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self._local_mesh, jax.sharding.PartitionSpec()
    )

  @functools.cached_property
  def _LOCAL_SHARDED(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
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
        and `kd.sharding.SHARDED` are supported.

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
    if sharding is self.SHARDED:
      array = jax.device_put(array, self._LOCAL_SHARDED)
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
      shardings: _ShardingValue,
  ) -> _T:
    """Like `jax.lax.with_sharding_constraint` but support forwarding sharding.

    Supports:

    * Better typing annotations
    * Support optional sharding

    Args:
      x: PyTree of jax.Arrays which will have their shardings constrained
      shardings: PyTree of sharding specifications. If `None`, `x` sharding is
        unmodified.

    Returns:
      The sharded array.
    """

    def _merge(s, x_):
      if s is None:  # No sharding provided, forward the original sharding
        return x_
      else:
        return jax.lax.with_sharding_constraint(x_, s)

    return jax.tree_map(
        _merge,
        shardings,
        x,
        is_leaf=lambda x: x is None,
    )

  Sharding = Sharding  # pylint: disable=invalid-name


sharding = _ShardingAPI()
