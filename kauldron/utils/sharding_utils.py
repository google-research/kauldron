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

from collections.abc import Callable, Iterator
import contextlib
import dataclasses
import functools
import typing
from typing import TypeVar

import jax
from jax._src import source_info_util
from kauldron.typing import PyTree  # pylint: disable=g-importing-member
from kauldron.utils import _jax
import numpy as np

if typing.TYPE_CHECKING:
  from kauldron.train import train_step  # pylint: disable=g-bad-import-order

# Sharding value can be:
# * None: Propagate current sharding
# * jax.sharding.Sharding: Use this sharding for all sub-tree
# * Callable: Lazily compute the sharding from the array sub-tree
ShardingTree = PyTree[
    None | jax.sharding.Sharding | Callable[[PyTree[jax.Array]], 'ShardingTree']
]
_T = TypeVar('_T')

# Some error messages raised by jax.jit display the code where
# `jax.lax.with_sharding_constraint` was called. To be more informative,
# we skip this util file so the error point to where
# `kd.sharding.with_sharding_constraint` was called.
source_info_util.register_exclusion(__file__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShardingStrategy:
  """Sharding strategy.

  This class defines the sharding for the dataset, params, optimizer, etc...

  Each sharding can be a PyTree with leafs being one of:

  * None: No sharding specified, `jax.jit` will auto-compute the sharding
  * jax.sharding.Sharding: Use this sharding for all sub-tree
  * Callable: Lazily compute the sharding from the array sub-tree
  """

  # TODO(epot): Rename to `batch` ?
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
    )

  @contextlib.contextmanager
  def set_global_mesh(self) -> Iterator[None]:
    """Might activate the mesh.

    Our implementation does nothing here, but some codebases/model
    implementations set the mesh globally so it can be used inside the model,
    rather than explicitly propagating it.

    Yields:
      None
    """
    yield


@dataclasses.dataclass(frozen=True, kw_only=True)
class FSDPSharding:
  """Simple FSDP-like sharding rule.

  Shards the largest dimension that is not sharded already and is divisible
  by the total device count.

  Usage:

  ```python
  cfg.sharding = kd.sharding.ShardingStrategy(
      params=kd.sharding.FSDPSharding(),
  )
  ```

  Attributes:
    min_size_to_shard_mb: minimal tensor size to bother with sharding.
  """

  min_size_to_shard_mb = 4

  def __call__(self, tree: PyTree[jax.Array]) -> ShardingTree:
    """Apply the sharding rule to the given tree."""
    # Implementation inspired from the big_vision codebase:
    # http://https://github.com/google-research/big_vision/tree/HEAD/big_vision/sharding.py;l=91;rcl=651681534
    min_size_to_shard_bytes = self.min_size_to_shard_mb * (2**20)

    def _apply(x: jax.Array) -> jax.sharding.Sharding:

      if _nbytes(x) <= min_size_to_shard_bytes:
        return sharding.REPLICATED

      # Partition along largest axis that is divisible and not taken.
      idx = np.argsort(x.shape)[::-1]

      spec = [None] * x.ndim
      for i in idx:
        if x.shape[i] % jax.device_count() == 0:
          spec[i] = 'devices'
          return jax.sharding.NamedSharding(
              sharding.DEVICES_MESH,  # pylint: disable=protected-access
              jax.sharding.PartitionSpec(*spec),
          )
      # TODO(epot): Should log params for weights non-divisible ?
      # At least some global XX arrays could not be sharded. and collect the
      # names
      return sharding.REPLICATED

    return jax.tree.map(_apply, tree)


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
  def DEVICES_MESH(self) -> jax.sharding.Mesh:  # pylint: disable=invalid-name
    """Default mesh, with a single axis containing all devices."""
    devices = np.asarray(jax.devices())
    return jax.sharding.Mesh(devices, axis_names=('devices',))

  @functools.cached_property
  def REPLICATED(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self.DEVICES_MESH, jax.sharding.PartitionSpec()
    )

  @functools.cached_property
  def FIRST_DIM(self) -> jax.sharding.NamedSharding:  # pylint: disable=invalid-name
    return jax.sharding.NamedSharding(
        self.DEVICES_MESH, jax.sharding.PartitionSpec('devices')
    )

  @functools.cached_property
  def _process_count(self) -> int:
    return jax.process_count()

  def device_put(self, arrays: _T, sharding: jax.sharding.NamedSharding) -> _T:  # pylint: disable=redefined-outer-name
    """Similar to `jax.device_put`, but support multi-process.

    Args:
      arrays: The nested tree of array to shard/replicate
      sharding: How to shard the array.

    Returns:
      The replicated nested tree of array
    """

    def _to_global_array(x):
      if isinstance(x, (int, float, bool)):  # Scalar support (e.g. `step`)
        x = np.asarray(x)

      # Contrary to `jax.make_array_from_single_device_arrays`,
      # `jax.make_array_from_process_local_data` trigger
      # `Disallowed host-to-device transfer`, so need to wrap it in a
      # `transfer_guard`
      with jax.transfer_guard('allow'):
        return jax.make_array_from_process_local_data(
            sharding,
            x,
            global_shape=_jax.local_to_global_shape(x.shape, sharding=sharding),
        )

    return jax.tree.map(_to_global_array, arrays)

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
      elif jax.tree.reduce(_all_shape_dtype_struct, x_, True):
        # If the tree is a `ShapeDtypeStruct`
        return jax.tree.map(
            lambda array: _jax.replace_shape_dtype_struct(array, sharding=s),
            x_,
        )
      else:
        return jax.lax.with_sharding_constraint(x_, s)

    return jax.tree.map(
        _merge,
        shardings,
        x,
        is_leaf=lambda x: x is None,
    )

  ShardingStrategy = ShardingStrategy  # pylint: disable=invalid-name

  # Typing annotation to annotate sharding pytree
  ShardingTree = ShardingTree  # pylint: disable=invalid-name

  # Sharding implementations:
  FSDPSharding = FSDPSharding  # pylint: disable=invalid-name


def _all_shape_dtype_struct(state, x):
  return state and isinstance(x, jax.ShapeDtypeStruct)


def _nbytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
  """Equivalent to `x.nbytes`, but support `jax.ShapeDtypeStruct`."""
  return int(np.prod(x.shape) * np.dtype(x.dtype).itemsize)


sharding = _ShardingAPI()
