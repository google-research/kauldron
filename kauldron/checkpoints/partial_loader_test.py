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
from kauldron.projects.examples import mnist_autoencoder
import pytest
import tensorflow_datasets as tfds

with kd.konfig.imports():
  from flax import linen as nn  # pylint: disable=g-import-not-at-top
  from kauldron import kd as kd_cfg  # pylint: disable=g-import-not-at-top,reimported

_NUM_FEATURES = 6


@pytest.fixture  # (scope='module')
def new_cfg(tmp_path: pathlib.Path) -> kd.train.Config:
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
def old_cfg(tmp_path: pathlib.Path):
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


def _make_chpt(
    old_cfg: kd.train.Config,  # pylint: disable=redefined-outer-name
    new_cfg: kd.train.Config,  # pylint: disable=redefined-outer-name
    old_to_new: dict[str, str],
):
  loader = kd.ckpts.PartialLoader(  # pytype: disable=wrong-arg-types
      source=kd.ckpts.KauldronSource(old_cfg.workdir),  # pylint: disable=missing-kwoa
      old_to_new=old_to_new,
  )

  ckpt = kd.train.Checkpointer(  # pytype: disable=wrong-arg-types
      workdir=new_cfg.workdir,
      save_interval_steps=1,
      fallback_loader=loader,
  )
  return ckpt


def test_loader(new_cfg: kd.train.Config, old_cfg: kd.train.Config):  # pylint: disable=redefined-outer-name
  old_state = old_cfg.init_state()
  old_cfg.checkpointer.save_state(old_state, 0)

  init_state = new_cfg.init_state()

  ckpt = _make_chpt(
      old_cfg,
      new_cfg,
      old_to_new={
          'params.encoder': 'params.decoder.layers_1',
      },
  )
  new_state = ckpt.restore(init_state)

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

  ckpt = _make_chpt(
      old_cfg,
      new_cfg,
      old_to_new={
          'params.encoder': 'params.encoder',
      },
  )
  new_state = ckpt.restore(init_state)
  chex.assert_trees_all_close(
      new_state.params['encoder'],
      old_state.params['encoder'],
  )


def _assert_not_all_close(*args):
  with pytest.raises(AssertionError):
    chex.assert_trees_all_close(*args)