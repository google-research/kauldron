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

from typing import Any
from unittest import mock

import jax
from kauldron.utils import _jax
import numpy as np
import pytest


class MockDevice:

  def __init__(self, device_id, process_index):
    self.id = device_id
    self.process_index = process_index
    self.host_id = process_index
    self.platform = 'TPU'
    self.device_kind = 'TPU v4'

  def __eq__(self, other):
    return isinstance(other, MockDevice) and self.id == other.id

  def __hash__(self):
    return hash(self.id)


@pytest.mark.parametrize(
    'mesh_shape, devices_per_process, axes, in_shape, expected',
    [
        # Format: mesh_shape (replica, batch, seq, model),
        # devices_per_process, axes, in_shape, expected
        # 1. Original tests (1 device per process, sharding only on batch)
        ((1, 1, 1, 1), 1, (), (3, 4), (3, 4)),
        ((1, 4, 1, 1), 1, (), (3, 4), (3, 4)),
        ((1, 1, 1, 1), 1, ('batch',), (3, 4), (3, 4)),
        ((1, 4, 1, 1), 1, ('batch',), (3, 4), (12, 4)),
        # 2. Multi-device per process (8 devices per process like TPU v4)
        # 8 devices total (1 process), replicated
        ((1, 1, 4, 2), 8, (), (3, 4), (3, 4)),
        # 8 devices total (1 process), sharded batch (batch axis shape 1,
        # so no sharding scale)
        ((1, 1, 4, 2), 8, ('batch',), (3, 4), (3, 4)),
        # 16 devices total (2 processes, 8 dev/proc),
        # mesh: (replica=1, batch=2, seq=4, model=2)
        # Data sharded by batch (2 partitions). Process 0 has 8 devices.
        # Process 0 devices: 0..7 (covers batch=0)
        # Process 1 devices: 8..15 (covers batch=1)
        # Scale factor = 2 // 1 = 2
        ((1, 2, 4, 2), 8, ('batch',), (3, 4), (6, 4)),
        # 32 devices total (4 processes, 8 dev/proc),
        # mesh: (replica=1, batch=4, seq=4, model=2)
        # Data sharded by batch (4 partitions).
        # Process 0 devices: 0..7 (covers batch=0)
        # Scale factor = 4 // 1 = 4
        ((1, 4, 4, 2), 8, ('batch',), (3, 4), (12, 4)),
        # 3. 2D sharding on first dimension (replica and batch both sharded)
        # mesh: (replica=2, batch=2, seq=2, model=2) = 16 devices,
        # 2 processes.
        # axes: (('replica', 'batch'), None)
        # Scale factor = 4 // 2 = 2
        ((2, 2, 2, 2), 8, (('replica', 'batch'), None), (3, 4), (6, 4)),
    ],
)
def test_local_to_global_shape(
    mesh_shape: tuple[int, int, int, int],
    devices_per_process: int,
    axes: tuple[Any, ...],
    in_shape: tuple[int, ...],
    expected: tuple[int, ...],
):
  total_devices = int(np.prod(mesh_shape))
  process_count = total_devices // devices_per_process
  if process_count == 0:
    process_count = 1

  devices = [
      MockDevice(i, i // devices_per_process) for i in range(total_devices)
  ]
  devices_arr = np.array(devices).reshape(mesh_shape)

  mesh = jax.sharding.Mesh(
      devices_arr,
      axis_names=('replica', 'batch', 'seq', 'model'),
  )

  sharding = jax.sharding.NamedSharding(
      mesh,
      spec=jax.sharding.PartitionSpec(*axes),
  )

  with mock.patch.object(
      jax, 'process_count', return_value=process_count
  ), mock.patch.object(jax, 'process_index', return_value=0):
    out_shape = _jax.local_to_global_shape(in_shape, sharding=sharding)
    assert out_shape == expected


def test_local_to_global_shape_fail():

  devices = np.array(jax.devices())
  devices = devices.reshape((1, 1, 1, 1))

  mesh = jax.sharding.Mesh(
      devices,
      axis_names=('replica', 'batch', 'seq', 'model'),
  )

  sharding = jax.sharding.NamedSharding(
      mesh,
      spec=jax.sharding.PartitionSpec('batch', 'seq'),
  )

  with pytest.raises(ValueError, match='Data can only be sharded on the first'):
    _jax.local_to_global_shape((3, 4), sharding=sharding)
