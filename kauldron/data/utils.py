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

"""Data Utils."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

from etils import enp
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
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

  def __post_init__(self):
    num_devices = jax.device_count()
    if self.total % num_devices != 0:
      raise ValueError(
          "batch_size must be divisible by num_devices."
          f" batch_size={self.total} {num_devices=}"
      )

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

  if spec.shape[0] is None:
    return jnp.empty((batch_dim,) + spec.shape[1:], dtype)
  else:
    return jnp.empty(spec.shape, dtype)


def mock_batch_from_elem_spec(
    elem_spec: ElementSpec, elem_sharding: sharding_utils.ShardingTree
) -> PyTree[jax.Array]:
  """Create a mock batch from the element_spec of a data iterator."""
  elem_spec = etree.spec_like(elem_spec)

  def _get_global_shape(spec):
    return ArraySpec(
        shape=_jax.local_to_global_shape(spec.shape, sharding=elem_sharding),
        dtype=spec.dtype,
    )

  elem_spec = jax.tree.map(_get_global_shape, elem_spec)

  @jax.jit
  def jitted_create_mock_batch():
    mock_batch = jax.tree.map(array_spec_to_jnp_empty, elem_spec)
    mock_batch = sharding_utils.sharding.with_sharding_constraint(
        mock_batch, elem_sharding
    )
    return mock_batch

  mock_batch = jitted_create_mock_batch()

  return mock_batch


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


def spec_to_json(spec: PyTree[enp.ArraySpec]) -> epy.typing.Json:
  """Convert the element_spec tree to a json representation."""
  # TODO(epot): Supports non-dict structures (dataclasses, and arbitrary
  # `__json__` protocol). Likely cannot use jax `xx_with_path` function to
  # automatically convert any tree to a json as the objects couldn't be
  # restored.
  return jax.tree.map(_spec_to_json, spec)


def _spec_to_json(spec: enp.ArraySpec) -> epy.typing.Json:
  return {
      "shape": spec.shape,
      "dtype": spec.dtype.name,
  }


def json_to_spec(data: epy.typing.Json) -> PyTree[enp.ArraySpec]:
  """Convert a json representation to a element_spec tree."""
  return jax.tree.map(_json_to_spec, data, is_leaf=_is_spec_leaf)


def _is_spec_leaf(spec: epy.typing.Json) -> bool:
  return isinstance(spec, dict) and set(spec) == {"shape", "dtype"}


def _json_to_spec(data: epy.typing.Json) -> enp.ArraySpec:
  return enp.ArraySpec(
      dtype=np.dtype(data["dtype"]),
      shape=data["shape"],
  )
