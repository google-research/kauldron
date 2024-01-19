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

"""Test."""

import pathlib

import chex
from kauldron import kd
from examples import mnist_autoencoder
from kauldron.utils import assert_utils
import pytest
import tensorflow_datasets as tfds

with kd.konfig.imports():
  from flax import linen as nn  # pylint: disable=g-import-not-at-top
  from kauldron import kd as kd_cfg  # pylint: disable=g-import-not-at-top,reimported

_NUM_FEATURES = 6


@pytest.fixture  # (scope='module')
def new_trainer(tmp_path: pathlib.Path) -> kd.train.Trainer:
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = str(tmp_path / 'new_workdir')

  cfg.model = kd_cfg.nn.FlatAutoencoder(
      inputs='batch.image',
      encoder=nn.Dense(features=_NUM_FEATURES),
      decoder=nn.Sequential([
          nn.Dense(features=784),
          nn.Dense(features=_NUM_FEATURES),
          nn.Dense(features=28 * 28),
      ]),
  )

  cfg.train_ds.loader.__qualname__ = 'kauldron.data.loaders.Tfds'
  cfg.train_ds.batch_size = 1
  cfg.seed = 0
  cfg = kd.konfig.resolve(cfg)
  with tfds.testing.mock_data():
    _ = cfg.train_ds.element_spec
  return cfg


@pytest.fixture  # (scope='module')
def old_trainer(tmp_path: pathlib.Path):
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = str(tmp_path / 'old_workdir')

  cfg.model = kd_cfg.nn.FlatAutoencoder(
      inputs='batch.image',
      encoder=nn.Dense(features=_NUM_FEATURES),
      decoder=nn.Dense(features=28 * 28),
  )

  cfg.train_ds.loader.__qualname__ = 'kauldron.data.loaders.Tfds'
  cfg.train_ds.batch_size = 1
  cfg.seed = 1
  cfg = kd.konfig.resolve(cfg)
  with tfds.testing.mock_data():
    _ = cfg.train_ds.element_spec
  return cfg


def _make_loader(
    old_trainer: kd.train.Trainer,  # pylint: disable=redefined-outer-name
    new_to_old: dict[str, str],
):
  loader = kd.ckpts.PartialLoader(  # pytype: disable=wrong-arg-types
      source=kd.ckpts.KauldronSource(old_trainer.workdir),  # pylint: disable=missing-kwoa
      new_to_old=new_to_old,
  )
  return loader


def test_loader(new_trainer: kd.train.Trainer, old_trainer: kd.train.Trainer):  # pylint: disable=redefined-outer-name
  old_state = old_trainer.init_state()
  old_trainer.checkpointer.save(old_state, step=0)
  old_trainer.checkpointer.wait_until_finished()

  init_state = new_trainer.init_state()

  loader = _make_loader(
      old_trainer,
      new_to_old={
          'params.decoder.layers_1': 'params.encoder',
      },
  )
  new_state = loader.transform(init_state)
  assert_utils.assert_trees_all_same_type(init_state, new_state)

  # New weights have been copied
  chex.assert_trees_all_close(
      new_state.params['decoder']['layers_1'],
      old_state.params['encoder'],
  )
  # Make sure weight were different
  _assert_not_all_close(
      init_state.params['decoder']['layers_1'],
      new_state.params['decoder']['layers_1'],
  )
  # Encoder is untouched, as well as layer 0
  chex.assert_trees_all_close(
      new_state.params['encoder'],
      init_state.params['encoder'],
  )
  chex.assert_trees_all_close(
      new_state.params['decoder']['layers_0'],
      init_state.params['decoder']['layers_0'],
  )

  loader = _make_loader(
      old_trainer,
      new_to_old={
          'params.encoder': 'params.encoder',
      },
  )
  new_state = loader.transform(init_state)
  chex.assert_trees_all_close(
      new_state.params['encoder'],
      old_state.params['encoder'],
  )


def _assert_not_all_close(*args):
  with pytest.raises(AssertionError):
    chex.assert_trees_all_close(*args)
