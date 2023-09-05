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

"""Data Utils."""

from __future__ import annotations

from etils.etree import jax as etree  # pylint: disable=g-importing-member
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util
from kauldron.typing import Array, ArraySpec, ElementSpec, PyTree  # pylint: disable=g-multiple-import,g-importing-member
from kauldron.utils import core
import numpy as np


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

  if spec.shape[0] is None:
    return jnp.empty((batch_dim,) + spec.shape[1:], dtype)
  else:
    return jnp.empty(spec.shape, dtype)


def drop_first_axis(arr: Array['d ...']) -> Array['...']:
  """Drop the first axis of a given array. Used for model.init()."""
  return arr[0]


def mock_batch_from_elem_spec(
    elem_spec: ElementSpec, drop_device_axis: bool = True
) -> PyTree[jax.Array]:
  """Create a mock batch from the element_spec of a data iterator."""
  elem_spec = etree.spec_like(elem_spec)
  mock_batch = jax.tree_util.tree_map(array_spec_to_jnp_empty, elem_spec)
  if drop_device_axis:
    mock_batch = jax.tree_util.tree_map(drop_first_axis, mock_batch)
  return mock_batch

Args = tuple[PyTree[jax.Array], ...]
Kwargs = dict[str, jax.Array]


def get_model_inputs(
    model: nn.Module, context: core.Context
) -> tuple[Args, Kwargs]:
  """Get the inputs for a top-level module from a batch."""
  if core.is_key_annotated(model):
    return (), core.resolve_kwargs(model, context, func=model.__call__)
  else:
    return (context.batch,), {}
