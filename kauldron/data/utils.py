# Copyright 2025 The kauldron Authors.
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

"""Data Utils."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron.train import context as context_lib
from kauldron.typing import ArraySpec, ElementSpec, PyTree  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import _jax
from kauldron.utils import sharding_utils
import numpy as np


@dataclasses.dataclass(frozen=True)
class BatchSize:
  """Batch size.

  Attributes:
    total: The total batch size across all hosts/devices
    per_process: The batch size for a single host
  """

  total: int

  @functools.cached_property
  def per_process(self) -> int:
    return self.total // jax.process_count()


def array_spec_to_jnp_empty(spec: ArraySpec, batch_dim: int = 17) -> jax.Array:
  """Convert a tf.data.TensorSpec element_spec to a jnp.empty array.

  Used for initializing a model.

  Args:
    spec: A TensorSpec. For example as returned by DatasetIterator.element_spec
    batch_dim: If the first dimension of TensorSpec is None it is replaced by
      this.

  Returns:
    An empty jnp array with the requested shape and dtype.
  """
  # silently convert int64 -> int32 to avoid jax warning
  dtype = jnp.int32 if spec.dtype in [jnp.int64, np.int64] else spec.dtype

  # Ensure spec.shape is fully concrete by replacing None with 1. This is a
  # workaround for datasets that yield variable shapes, as JAX requires concrete
  # shapes for jnp.empty and JIT compilation. The mock batch created with these
  # placeholder dimensions might not accuratelyrepresent all variable data.
  if spec.shape[0] is None:
    # First dimension is dynamic (e.g. batch size), use batch_dim. Make sure the
    # rest of the dimensions are also concrete by replacing None with 1.
    remaining_dims = _make_concrete_shape(spec.shape[1:])
    final_shape = (batch_dim,) + remaining_dims
    return jnp.empty(final_shape, dtype)
  else:
    # All dimensions defined in spec.shape, make them all concrete. This branch
    # is hit if spec.shape is e.g. (B, None, H, W) or in your case (1, None,
    # None). Replace any None with 1.
    final_shape = _make_concrete_shape(spec.shape)
    return jnp.empty(final_shape, dtype)


def mock_batch_from_elem_spec(
    elem_spec: ElementSpec, elem_sharding: sharding_utils.ShardingTree
) -> PyTree[jax.Array]:
  """Create a mock batch from the element_spec of a data iterator."""
  elem_spec = get_global_elem_spec(elem_spec, elem_sharding)

  @jax.jit
  def jitted_create_mock_batch():
    mock_batch = jax.tree.map(array_spec_to_jnp_empty, elem_spec)
    mock_batch = sharding_utils.sharding.with_sharding_constraint(
        mock_batch, elem_sharding
    )
    return mock_batch

  mock_batch = jitted_create_mock_batch()

  return mock_batch


def get_global_elem_spec(
    per_host_elem_spec: ElementSpec, elem_sharding: sharding_utils.ShardingTree
) -> ElementSpec:
  """Converts the per-host element spec to a global element spec."""

  def _get_global_shape(spec):
    return ArraySpec(
        shape=_jax.local_to_global_shape(spec.shape, sharding=elem_sharding),
        dtype=spec.dtype,
    )

  return jax.tree.map(_get_global_shape, per_host_elem_spec)


Args = tuple[PyTree[jax.Array], ...]
Kwargs = dict[str, jax.Array]


def get_model_inputs(
    model: nn.Module, context: context_lib.Context
) -> tuple[Args, Kwargs]:
  """Get the inputs for a top-level module from a batch."""
  if kontext.is_key_annotated(model):
    return (), kontext.resolve_from_keyed_obj(
        context, model, func=model.__call__
    )
  else:
    return (context.batch,), {}


def get_model_inputs_from_batch(
    model: nn.Module,
    batch: ElementSpec,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Returns dummy (args, kwargs) to pass to the model input.

  Args:
    model: Flax model
    batch: The batch structure from which extract the inputs

  Returns:
    args: The positional arguments
    kwargs: The keyword arguments
  """
  context = context_lib.Context(step=0, batch=batch)
  args, kwargs = get_model_inputs(model, context)
  return args, kwargs


def _make_concrete_shape(shape: tuple[int | None, ...]) -> tuple[int, ...]:
  """Replaces any None in a shape with 1."""
  return tuple(d if d is not None else 1 for d in shape)
