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

"""Jax utils."""

from collections.abc import Callable
from typing import ParamSpec, TypeVar
import jax
import numpy as np

_T = TypeVar('_T')
_P = ParamSpec('_P')


# TODO(epot): Remove this function once Jax provide this natively.
def eval_shape_with_sharding(
    fn: Callable[_P, _T],
    *args: _P.args,
    **kwargs: _P.kwargs,
) -> _T:
  """Like `jax.eval_shape`, but also auto-infer the sharding."""
  fn = jax.jit(fn)
  out = fn.eval_shape(*args, **kwargs)
  sharding = fn.lower(*args, **kwargs).compile().output_shardings

  return jax.tree.map(
      lambda x, s: replace_shape_dtype_struct(x, sharding=s),
      out,
      sharding,
  )


# TODO(epot): Should instead be a `jax.ShapeDtypeStruct.replace` method
def replace_shape_dtype_struct(
    obj: jax.ShapeDtypeStruct, **kwargs
) -> jax.ShapeDtypeStruct:
  """Like `dataclasses.replace`, but for `ShapeDtypeStruct`."""
  new_kwargs = {
      'shape': obj.shape,
      'dtype': obj.dtype,
      'sharding': obj.sharding,
      'weak_type': obj.weak_type,
  }
  new_kwargs.update(kwargs)
  return jax.ShapeDtypeStruct(**new_kwargs)


def local_to_global_shape(
    shape: tuple[int, ...],
    *,
    sharding: jax.sharding.Sharding,
) -> tuple[int, ...]:
  """Convert a per-process shape to a global shape."""
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise ValueError(
        f'Only NamedSharding is supported for now. Got: {sharding!r}'
    )

  if not sharding.spec:
    return shape

  if any(x is not None for x in sharding.spec[1:]):
    raise ValueError(
        f'Data can only be sharded on the first dimension. Got: {sharding!r}.'
    )

  # Calculate total partitions for the first dimension
  axis_names = sharding.spec[0]
  if axis_names is None:
    total_partitions = 1
  elif isinstance(axis_names, str):
    total_partitions = sharding.mesh.shape[axis_names]
  else:
    total_partitions = 1
    for name in axis_names:
      total_partitions *= sharding.mesh.shape[name]

  # Calculate local partitions for the first dimension on the current process
  process_index = jax.process_index()
  local_devices = [
      d for d in sharding.mesh.devices.flat if d.process_index == process_index
  ]

  if not local_devices:
    local_partitions = 1
  else:
    mesh_devices = sharding.mesh.devices
    axis_names_list = list(sharding.mesh.shape.keys())

    unique_axis_values = set()
    for device in local_devices:
      indices = np.where(mesh_devices == device)
      coord = {name: indices[i][0] for i, name in enumerate(axis_names_list)}

      if axis_names is None:
        val = 0
      elif isinstance(axis_names, str):
        val = coord[axis_names]
      else:
        val = tuple(coord[name] for name in axis_names)
      unique_axis_values.add(val)
    local_partitions = len(unique_axis_values)

  scaling_factor = total_partitions // local_partitions

  return (shape[0] * scaling_factor,) + shape[1:]
