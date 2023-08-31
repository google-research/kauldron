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
  """Jax sharding."""

  @functools.cached_property
  def _devices(self) -> np.ndarray:
    # mesh_utils.create_device_mesh((len(jax.devices()),))
    return np.asarray(jax.devices())

  @functools.cached_property
  def _device_mesh(self) -> jax.sharding.Mesh:
    return jax.sharding.Mesh(self._devices, axis_names=('devices',))

  @functools.cached_property
  def REPLICATED(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self._device_mesh, jax.sharding.PartitionSpec()
    )

  @functools.cached_property
  def SHARDED(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self._device_mesh, jax.sharding.PartitionSpec('devices')
    )

  @functools.cached_property
  def DEVICE0(self) -> jax.sharding.SingleDeviceSharding:  # pylint: disable=invalid-name
    return jax.sharding.SingleDeviceSharding(self._devices[0])

  # Should create corresponding methods `.replicate()`, `.shard()`,... ?
  # def tree_replicate(state: _T) -> _T:
  #   """Replicated sharding across devices."""
  #   return jax.tree_map(lambda x: jax.device_put(x, REPLICATED), state)


sharding = _Sharding()
