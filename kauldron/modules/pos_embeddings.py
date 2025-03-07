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

"""Position embedding layers."""

from __future__ import annotations

import warnings

from flax import linen as nn
import jax.numpy as jnp
from kauldron.modules import knn_types
from kauldron.typing import Axes, DType, Float, Initializer, Shape, typechecked  # pylint: disable=g-multiple-import,g-importing-member


class AddEmbedding(nn.Module):
  """Helper Module for adding a PositionEmbedding e.g. in a `knn.Sequential`.

  Attributes:
    emb: The position embedding to be added to the inputs.
    axis: The axis parameter passed to the position embedding for determining
      its shape. Usually set to -2, to get embeddings of shape `n d` for inputs
      of dimension `*b n d`.
  """

  emb: knn_types.PositionEmbedding
  axis: Axes

  def __call__(self, inputs: Float['*any']) -> Float['*any']:
    return inputs + self.emb(inputs.shape, axis=self.axis)


class FourierEmbedding(nn.Module):
  """Apply Fourier position embedding to a grid of coordinates.

  Attributes:
    num_fourier_bases: The number of Fourier bases to use. The embedding
      dimensionality is 2 x len(axis) x num_fourier_bases, but the result will
      be projected to match the given shape.

  Return:
    Fourier position embeddings broadcastable to given shape.
  """

  num_fourier_bases: int

  @typechecked
  @nn.compact
  def __call__(self, shape: Shape, *, axis: Axes) -> Float['...']:
    emb_shape = _get_embedding_shape_from_axes(shape, axis)
    coord_shape = emb_shape[:-1]  # skip feature axis

    # NeRF-style Fourier/sinusoidal position encoding.
    coords = _create_gradient_grid(coord_shape, value_range=(-jnp.pi, jnp.pi))
    pos_embedding = convert_to_fourier_features(
        coords, basis_degree=self.num_fourier_bases
    )
    # Project to desired feature dims.
    projected_pos_emb = nn.Dense(shape[-1], name='dense_pe')(pos_embedding)
    return jnp.broadcast_to(projected_pos_emb, shape=shape)


class LearnedEmbedding(nn.Module):
  """Learned positional embeddings.

  Implements the knn_types.PositionEmbedding protocol.

  Attributes:
    emb_init: Initializer for the position embeddings.
    dtype: DType of the position embedding. Default to float32.
  """

  dtype: DType = jnp.float32
  emb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @typechecked
  @nn.compact
  def __call__(self, shape: Shape, *, axis: Axes) -> Float['...']:
    """Return learned positional embeddings broadcast to shape.

    Args:
      shape: The shape of the output. Typically something like `tokens.shape`.
      axis: Which axes to use for the shape of the learned parameters.

    Returns:
      Learned position embeddings broadcast to given shape.
    """
    emb_shape = _get_embedding_shape_from_axes(shape, axis)
    pe = self.param('embeddings', self.emb_init, emb_shape, self.dtype)
    return jnp.broadcast_to(pe, shape)


class ZeroEmbedding(nn.Module):
  """Embedding that returns zero (for deactivating position embeddings).

  Implements the knn_types.PositionEmbedding protocol.
  """

  dtype: DType = jnp.float32

  @typechecked
  @nn.compact
  def __call__(self, shape: Shape, *, axis: Axes = ()) -> Float['...']:
    del axis
    return jnp.zeros(shape=shape, dtype=self.dtype)


class AddLearnedEmbedding(nn.Module):
  """Adds learned positional embeddings to the inputs.

  DEPRECATED: This module is deprecated in favor of the new LearnedEmbedding.

  Attributes:
    emb_init: Positional embedding initializer.
    axes: One or more axes which to include into the embedding shape. The
      feature axis (-1) is automatically included and should not be passed
      explicitly.

  Returns:
    Array with same shape as input.
  """

  emb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.
  axes: Axes = (-2, -1)

  @typechecked
  @nn.compact
  def __call__(self, inputs: Float['*b n d']) -> Float['*b n d']:
    warnings.warn(
        f'{self.__class__.__name__} is deprecated in favor of LearnedEmbedding',
        DeprecationWarning,
        stacklevel=2,
    )
    emb_shape = _get_shape_from_axes(inputs.shape, self.axes)
    # Squeeze embedding shape to the minimum, because Flax can't handle
    # broadcasting.
    squeezed_emb_shape = _lsqueeze_shape(emb_shape)
    pe = self.param(
        'embeddings', self.emb_init, squeezed_emb_shape, inputs.dtype
    )
    return inputs + pe


def _get_embedding_shape_from_axes(full_shape: Shape, axis: Axes) -> Shape:
  """Produces a broadcastable shape from given axes.

  Automatically includes the feature axis (-1)

  Args:
    full_shape: The full shape of the representation / tokens from which to
      derive the broadcastable embedding shape.
    axis: One or more axes which to include into the embedding shape. The
      feature axis (-1) is automatically included and should not be passed
      explicitly.

  Returns:
    A that is broadcastable to full_shape and has entries > 1 only for the
    specified axis, and without leading ones.
  """
  if isinstance(axis, int):
    axis = (axis,)

  if max(axis) >= 0:
    raise ValueError(f'All axis must be negative. Provided {axis=}.')
  if -1 in axis:
    raise ValueError(
        'Do not explicitly include feature axis (-1), as it is implicitly'
        f' included. Provided {axis=}.'
    )
  axis = axis + (-1,)

  shape = [1] * len(full_shape)
  for ax in axis:
    shape[ax] = full_shape[ax]

  # Squeeze shape to the minimum, i.e. remove all leading 1 dimensions to make
  # the embedding usable with arbitrary batch dimensions.
  while shape and shape[0] == 1:
    shape = shape[1:]
  return tuple(shape)


def _lsqueeze_shape(shape: Shape) -> Shape:
  """Removes all leading 1-dimensions from given shape. DEPRECATED."""
  shape = list(shape)
  while shape and shape[0] == 1:
    shape = shape[1:]
  return tuple(shape)


def _get_shape_from_axes(full_shape: Shape, axes: Axes) -> Shape:
  """Produces a broadcastable shape from given axes. DEPRECATED."""
  if isinstance(axes, int):
    axes = [axes]
  shape = [1] * len(full_shape)
  for ax in axes:
    shape[ax] = full_shape[ax]
  return tuple(shape)


@typechecked
def _create_gradient_grid(
    samples_per_dim: tuple[int, ...],
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> Float['...']:
  """Creates a tensor with equidistant entries from -1 to +1 in each dim.

  Args:
    samples_per_dim: Number of points to have along each dimension.
    value_range: In each dimension, points will go from range[0] to range[1]

  Returns:
    A tensor of shape samples_per_dim + (N,) where N is the number of entries
    in samples_per_dim that are bigger than 1.
  """
  effective_dims = [s for s in samples_per_dim if s != 1]
  s = [jnp.linspace(value_range[0], value_range[1], n) for n in effective_dims]
  grid = jnp.stack(jnp.meshgrid(*s, sparse=False, indexing='ij'), axis=-1)
  return grid.reshape(samples_per_dim + grid.shape[-1:])


@typechecked
def convert_to_fourier_features(
    inputs: Float['... D'], basis_degree: int
) -> Float['... d']:
  """Convert inputs to Fourier features, e.g. for positional encoding."""

  # inputs.shape = (..., n_dims).
  # inputs should be in range [-pi, pi] or [0, 2pi].
  n_dims = inputs.shape[-1]

  # Generate frequency basis.
  freq_basis = jnp.concatenate(  # shape = (n_dims, n_dims * basis_degree)
      [2**i * jnp.eye(n_dims) for i in range(basis_degree)], 1
  )

  # x.shape = (..., n_dims * basis_degree)
  x = inputs @ freq_basis  # Project inputs onto frequency basis.

  # Obtain Fourier features as [sin(x), cos(x)] = [sin(x), sin(x + 0.5 * pi)].
  return jnp.sin(jnp.concatenate([x, x + 0.5 * jnp.pi], axis=-1))
