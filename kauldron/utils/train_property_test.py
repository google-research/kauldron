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

"""."""

from flax import linen as nn
import jax.numpy as jnp
from kauldron.utils import train_property
import pytest


class MyModel(nn.Module):
  is_training = train_property.train_property()

  @nn.compact
  def __call__(self, x):
    if self.is_training:
      return (x, x)
    else:
      return (x, x, x)


def test_model():
  model = MyModel()
  x = jnp.zeros((3,))

  with pytest.raises(ValueError, match='`is_training=` kwargs was not set'):
    model.init({}, x)

  params = model.init({}, x, is_training=True)

  vals = model.apply(params, x, is_training=True)
  assert len(vals) == 2

  vals = model.apply(params, x, is_training=False)
  assert len(vals) == 3


def test_model_hash():
  model = MyModel()

  default_model = hash(model)
  assert default_model == hash(model)

  with train_property._set_training(True):
    train_model = hash(model)
    assert train_model == hash(model)

  with train_property._set_training(False):
    eval_model = hash(model)
    assert eval_model == hash(model)

  assert eval_model != train_model
  assert eval_model != default_model
  assert train_model != default_model
  assert default_model == hash(model)
