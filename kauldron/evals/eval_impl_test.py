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

"""Test."""

from collections.abc import Iterator
import os
from unittest import mock

from etils import epath
from kauldron import kd
from examples import mnist_autoencoder
import tensorflow_datasets as tfds


def test_eval_impl(tmp_path: epath.Path):
  # Load config and reduce size
  cfg = mnist_autoencoder.get_config()

  cfg.train_ds.batch_size = 2
  cfg.evals.eval.ds.batch_size = 1  # pytype: disable=attribute-error
  cfg.model.encoder.features = 3
  cfg.num_train_steps = 1
  cfg.workdir = os.fspath(tmp_path)

  with kd.konfig.mock_modules():
    for ev in cfg.evals.values():
      ev.run = kd.evals.StandaloneEveryCheckpoint()

  trainer = kd.konfig.resolve(cfg)

  def _mocked_iterator(**kwargs) -> Iterator[int]:
    del kwargs

    # Simulate train saving a step
    state = trainer.init_state()
    trainer.checkpointer.save(state, step=0)
    trainer.checkpointer.wait_until_finished()
    yield 0

  # Launch train
  with (
      tfds.testing.mock_data(),
      mock.patch(
          'orbax.checkpoint.checkpoint_utils.checkpoints_iterator',
          _mocked_iterator,
      ),
  ):
    aux = trainer.continuous_eval('eval')
    # Ensure at least one checkpoint was computed
    assert 'recon' in aux['eval'].loss_states
