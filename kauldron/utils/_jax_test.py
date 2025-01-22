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

from unittest import mock

import jax
from kauldron.utils import _jax
import numpy as np
import pytest


@pytest.mark.parametrize(
    'axes, process_count, in_shape, expected',
    [
        ((), 1, (3, 4), (3, 4)),
        ((), 4, (3, 4), (3, 4)),
        (('batch',), 1, (3, 4), (3, 4)),
        (('batch',), 4, (3, 4), (12, 4)),
        ((('batch', 'replica'),), 1, (3, 4), (3, 4)),
        ((('batch', 'replica'),), 4, (3, 4), (12, 4)),
    ],
)
def test_local_to_global_shape(
    axes: tuple[str, ...],
    process_count: int,
    in_shape: tuple[int, ...],
    expected: tuple[int, ...],
):
  devices = np.array(jax.devices())
  devices = devices.reshape((1, 1, 1, 1))

  mesh = jax.sharding.Mesh(
      devices,
      axis_names=('replica', 'batch', 'seq', 'model'),
  )

  sharding = jax.sharding.NamedSharding(
      mesh,
      spec=jax.sharding.PartitionSpec(*axes),
  )

  with mock.patch.object(jax, 'process_count', return_value=process_count):
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
