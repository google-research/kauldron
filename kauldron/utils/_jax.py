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

"""Jax utils."""

from collections.abc import Callable
from typing import ParamSpec, TypeVar
import jax

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
  """Convert a per-process shape to a global shape.

  Contrary to the jax version, this function always scale the sharded dimension
  by the number of processes.

  Example:

  * shape=(x, y), sharding=('a') -> (x * jax.process_count(), y)

  Args:
    shape: The local shape
    sharding: The sharding to apply

  Returns:
    The global shape
  """
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise ValueError(
        f'Only NamedSharding is supported for now. Got: {sharding!r}'
    )

  # TODO(epot): We only support sharding on the first dimension for now. To
  # supports arbitrary axes, the sharding would have to specify on which
  # dimensions are split across hosts and which are simply sharded
  match len(sharding.spec):
    case 0:
      return shape
    case 1:
      return (shape[0] * jax.process_count(),) + shape[1:]
    case _:
      # TODO(epot): Supports spec = `('batch', None)`
      raise ValueError(
          f'Data can only be sharded on the first dimension. Got: {sharding!r}.'
          ' Please raise an issue if you need this.'
      )
