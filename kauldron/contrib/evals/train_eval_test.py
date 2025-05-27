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

"""Test train_eval."""

import dataclasses
import os

from etils import epath
from flax import linen as nn
import jax
from jax import numpy as jnp
from kauldron import kd
from kauldron.contrib.evals import train_eval as train_eval_lib
from examples.contrib import train_evaluator as train_eval_config
import tensorflow_datasets as tfds


def test_train_eval(tmp_path: epath.Path):
  """Test train_eval."""
  cfg = train_eval_config.get_config()
  cfg.workdir = os.fspath(tmp_path)
  cfg.stop_after_steps = 1

  readout_cfg = cfg.evals['readout'].readout_config  # pytype: disable=attribute-error
  readout_cfg.num_train_steps = 1
  kd.kontext.set_by_path(readout_cfg, '**.batch_size', 1)
  kd.kontext.set_by_path(readout_cfg, 'evals.*.num_batches', 1)

  with kd.konfig.mock_modules():
    cfg.model.encoder = nn.Dense(features=2)  # Smaller network.

  trainer = kd.konfig.resolve(cfg)
  with tfds.testing.mock_data(num_examples=10):
    state_specs = trainer.state_specs

    state = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), state_specs)

    train_evaluator = trainer.evals['readout']
    assert isinstance(train_evaluator, kd.contrib.evals.TrainEvaluator)
    aux = train_evaluator.evaluate(state, 0)

    readout_trainer = train_evaluator.readout_trainer_for_step(state, 0)  # pytype: disable=attribute-error
    # Sanity check to check that the `init_transform` are correctly propagated
    # (as the root `init_transform` is mutated, we should make sure the
    # changes are reflected on the `trainstep`)
    assert (
        readout_trainer.init_transform
        is readout_trainer.trainstep.init_transform
    )
    assert isinstance(
        readout_trainer.evals['eval'].writer, train_eval_lib._EvalWriter
    )
    # Assert mutating the `init_transform` do not change the hash/eq (so it do
    # not trigger new `jit` compilations).
    overwrite_transform = readout_trainer.init_transform
    old_hash = hash(readout_trainer.init_transform)
    overwrite_transform = dataclasses.replace(
        overwrite_transform,
        state_from_training=jax.tree.map(
            lambda x: jnp.ones(x.shape, x.dtype), state_specs
        ),
    )
    assert hash(overwrite_transform) == old_hash
    del aux
