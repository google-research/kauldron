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

"""Test."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses

import chex
from etils.array_types import f32
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron import klinen as knn
from kauldron import random as krandom
import numpy as np
import pytest

_IN_SHAPE = (3, 2)

# TODO(epot): Test when mixing `knn` inside `nn` modules (both attribute and
# nested attribute (e.g. Sequential(childs=)))
# TODO(epot): Test when having `nn` module as attribute of `knn`


class Nested(knn.Module):
  """Nested module."""

  child: knn.Module

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return self.child(x)


class ModelRoot(knn.Module):
  """Root model."""

  # Module can be:
  # * Root
  # * Attribute
  # * Nested attribute
  # * Compact
  child: knn.Module
  childs: list[knn.Module]
  hidden: Callable[[], knn.Module]
  nested: Nested

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    x = self.child(x)
    for child in self.childs:
      x = child(x)
    x = self.hidden()(x)
    x = self.nested(x)
    return x


class DenseAndDropout(knn.Module):
  """Simple model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(2)(x)
    x = nn.Dropout(0.1)(x, deterministic=not self.training)
    return x


class WithIntermediate(knn.Module):
  """Simple model."""

  tmp_val: knn.Intermediate[jax.Array] = dataclasses.field(init=False)
  tmp_list: knn.Intermediate[list[jax.Array]] = dataclasses.field(
      init=False, default_factory=list
  )

  @nn.compact
  def __call__(self, x):
    self.tmp_list.append(x)

    if self.name != 'childs_1':  # Sharded intermediates are shared between call
      # TODO(epot): Why the explicitly given `name='shared'` is lost here. Flax
      # bug ?
      with pytest.raises(
          AttributeError, match='Attribute was not set during the call'
      ):
        _ = self.tmp_val

    # Overwrite tmp_val
    self.tmp_val = x
    self.tmp_val = nn.Dense(2)(self.tmp_val)

    self.tmp_list.append(x)
    return self.tmp_val


def _make_model(module_cls: type[knn.Module]) -> ModelRoot:
  """Model factory."""
  shared = module_cls(name='shared')
  model_raw = ModelRoot(
      child=module_cls(),
      childs=[
          module_cls(),
          shared,
          shared,
      ],
      hidden=lambda: module_cls(),  # pylint: disable=unnecessary-lambda
      nested=Nested(module_cls()),
  )

  rng = jax.random.PRNGKey(0)

  return model_raw.init_bind(rng, f32[(*_IN_SHAPE,)])


def test_non_bind():
  model_raw = DenseAndDropout()

  with pytest.raises(ValueError, match='before calling'):
    _ = model_raw.params  # Unbind, function not available

  with pytest.raises(flax.errors.CallCompactUnboundModuleError):
    model_raw(jnp.ones(_IN_SHAPE))


def test_train_mode():
  model = _make_model(DenseAndDropout)

  model = model.with_rng(0)

  model_train = model
  assert model.training
  assert model.child.training
  assert model.childs[0].training
  assert model.childs[1].training
  assert model.childs[2] is model.childs[1]
  assert model.nested.training
  assert model.nested.child.training

  x = jnp.ones(_IN_SHAPE)
  y = model(x)
  assert not np.allclose(y, x)  # Dropout applied
  assert not np.allclose(y[0], y[1])  # Batch have different dropout
  y2 = model(x)
  np.testing.assert_allclose(y, y2)  # Calling model twice yield same result

  model = model.eval()
  assert not model.child.training
  assert not model.childs[0].training
  assert not model.childs[1].training
  assert model.childs[2] is model.childs[1]
  assert model.childs[2] is not model_train.childs[1]
  assert not model.nested.training
  assert not model.nested.child.training

  # Dropout disabled: No-op
  y = model(x)
  assert not np.allclose(y, y2)  # Dropout disabled
  # In eval, same example yield same result
  np.testing.assert_allclose(y[0], y[1])

  nested = model.nested.train()
  # Model isn't mutated
  assert not model.child.training
  assert not model.childs[0].training
  assert not model.childs[1].training
  assert not model.nested.training
  assert not model.nested.child.training
  # But nested is updated
  assert nested.training
  assert nested.child.training


def test_rng():
  x = jnp.ones(_IN_SHAPE)
  model = _make_model(DenseAndDropout)

  with pytest.raises(flax.errors.InvalidRngError):
    model(x)  # No rng by default

  assert model.rngs == {}  # pylint: disable=g-explicit-bool-comparison
  assert model.child.rngs == {}  # pylint: disable=g-explicit-bool-comparison
  assert model.childs[0].rngs == {}  # pylint: disable=g-explicit-bool-comparison

  key = krandom.PRNGKey(0)
  x = jnp.ones(_IN_SHAPE)

  model = model.with_rng(key)

  y = model(x)
  y2 = model(x)
  np.testing.assert_allclose(y, y2)

  # Test with_rng
  model = model.with_rng()  # Next key
  y = model(x)
  assert not jnp.allclose(y, y2)  # Old pred is different from the new key
  np.testing.assert_allclose(y, model(x))

  # rng is constant between train/eval
  jax.tree.map(
      np.testing.assert_allclose, model.rngs, model.eval().rngs
  )

  # Key can be explicitly passed
  model = model.with_rng({'dropout': key})
  y = model(x)
  y2 = model(x)
  np.testing.assert_allclose(y, y2)


def test_jit():
  @jax.jit
  def fn(model, x):
    y = model(x)
    return y

  x = jnp.ones(_IN_SHAPE)
  model = _make_model(DenseAndDropout)
  model = model.with_rng(0)

  y = fn(model, x)
  y2 = model(x)
  np.testing.assert_allclose(y, y2, atol=1e-6)

  new_y = fn(model.with_rng(), x)
  new_y2 = model.with_rng()(x)
  np.testing.assert_allclose(new_y, new_y2, atol=1e-6)

  # Calling jit with new rng should yield new results
  assert not np.allclose(new_y, y, atol=1e-6)


def test_param():
  model = _make_model(DenseAndDropout)
  model = model.with_rng(0)

  assert isinstance(model.params, flax.core.FrozenDict)
  assert isinstance(model.child.params, flax.core.FrozenDict)

  # Values are as expected
  chex.assert_trees_all_close(
      model.nested.params,
      flax.core.FrozenDict({'child': model.nested.child.params}),
  )


def test_intermediate():
  model = _make_model(WithIntermediate)

  x = jnp.ones(_IN_SHAPE)

  y = model(x)  # Model call works without intermediate

  with pytest.raises(
      AttributeError, match='can only be accessed inside module functions'
  ):
    _ = model.child.tmp_val

  with pytest.raises(
      AttributeError, match='can only be accessed inside module functions'
  ):
    _ = model.child.tmp_list

  with model.capture_intermediates() as intermediates:
    y2 = model(x)

  np.testing.assert_allclose(y, y2)
  # Last called captured value should match output
  np.testing.assert_allclose(y, intermediates.nested.child.tmp_val)
  # But not the first one
  _assert_not_all_close(y, intermediates.child.tmp_val)

  assert len(intermediates.child.tmp_list) == 2
  assert len(intermediates.childs[0].tmp_list) == 2
  assert len(intermediates.childs[1].tmp_list) == 4  # Shared called twice

  # Calling intermediate twice should reset the values
  with model.capture_intermediates() as intermediates:
    y2 = model(x)

  np.testing.assert_allclose(y, y2)
  # Last called captured value should match output
  np.testing.assert_allclose(y, intermediates.nested.child.tmp_val)
  # But not the first one
  _assert_not_all_close(y, intermediates.child.tmp_val)

  assert len(intermediates.child.tmp_list) == 2
  assert len(intermediates.childs[0].tmp_list) == 2
  assert len(intermediates.childs[1].tmp_list) == 4  # Shared called twice


def _assert_not_all_close(x, y):
  assert not np.allclose(x, y)
