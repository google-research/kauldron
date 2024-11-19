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

import os

from etils import epath
from etils.etree import jax as etree  # pylint: disable=g-importing-member
from jax import numpy as jnp
from kauldron import kd
from examples import mnist_autoencoder
import tensorflow_datasets as tfds


def test_sharding(tmp_path: epath.Path):
  # Load config and reduce size
  cfg = mnist_autoencoder.get_config()

  cfg.train_ds.batch_size = 1
  cfg.model.encoder.features = 3
  cfg.workdir = os.fspath(tmp_path)

  trainer = kd.konfig.resolve(cfg)

  # Get the state
  with tfds.testing.mock_data():
    state = trainer.init_state()

  del state
  # TODO(epot): How to test this is actually working?
  # for k, v in kd.kontext.flatten_with_path(state).items():
  #   assert v.sharding == kd.sharding.REPLICATED, k


def test_with_sharding_constraint():

  def _shard(x):
    etree.backend.assert_same_structure(x, {'inner': 0})
    return kd.sharding.REPLICATED

  x = {
      'a': {'inner': jnp.ones((1, 2))},
      'b': {'inner': jnp.ones((2, 3))},
      'c': {'inner': jnp.ones((2, 3))},
  }
  sharding = {
      'a': kd.sharding.REPLICATED,
      'b': None,
      'c': _shard,
  }
  sharded_x = kd.sharding.with_sharding_constraint(x, sharding)

  etree.backend.assert_same_structure(sharded_x, x)

  # TODO(epot): Test the actual values (should mock the devices)
