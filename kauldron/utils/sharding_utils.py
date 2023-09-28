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

"""Jax sharding utils."""

import functools
from typing import TypeVar

import jax
import numpy as np

_T = TypeVar('_T')


# Use class so methods are lazily called. Otherwise, `jax.devices` fail because
# `app.run(main)` is not yet called
class _Sharding:
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
        array.device_buffers,
    )


sharding = _Sharding()
