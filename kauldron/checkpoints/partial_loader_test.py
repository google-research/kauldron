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

import pathlib

import chex
from kauldron import kd
from examples import mnist_autoencoder
from kauldron.testing import assert_utils
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
  loader = kd.ckpts.PartialKauldronLoader(  # pytype: disable=wrong-arg-types
      workdir=old_trainer.workdir,  # pylint: disable=missing-kwoa
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


def test_():
  other, opt_state = kd.ckpts.partial_loader._split_new_to_old({
      'step': 'step',
      'params': 'params',
      'collections': 'collections',
      'opt_state.inner_opt_state': 'opt_state.inner_opt_state',
      'opt_state.mini_step': 'opt_state.mini_step',
      'opt_state.gradient_step': 'opt_state.gradient_step',
  })
  assert other == {
      'step': 'step',
      'params': 'params',
      'collections': 'collections',
  }
  assert opt_state == {
      'opt_state.inner_opt_state': 'opt_state.inner_opt_state',
      'opt_state.mini_step': 'opt_state.mini_step',
      'opt_state.gradient_step': 'opt_state.gradient_step',
  }


class TestTransformFn:
  """Tests for _PartialRestoreCheckpointItem._transform_fn."""

  def _make_item(
      self,
      state,
      new_to_old,
      ignore_model_keys=(),
      ignore_restored_keys=(),
  ):
    return kd.ckpts.partial_loader._PartialRestoreCheckpointItem(
        state=state,
        new_to_old=new_to_old,
        ignore_model_keys=ignore_model_keys,
        ignore_restored_keys=ignore_restored_keys,
    )

  def test_matching_structure(self):
    """When structures match, pass through unchanged."""
    state = {'params': {'a': 1, 'b': 2}}
    restored = {'old_params': {'a': 10, 'b': 20}}
    item = self._make_item(state, {'params': 'old_params'})
    result = item._transform_fn(restored)
    assert result == {'params': {'a': 10, 'b': 20}}

  def test_extra_keys_in_restored_ignored(self):
    """Extra keys in restored are removed when matched by ignore_restored_keys."""
    state = {'params': {'a': 1}}
    restored = {'old_params': {'a': 10, 'b': 20}}
    item = self._make_item(
        state,
        {'params': 'old_params'},
        ignore_restored_keys=('**.b',),
    )
    result = item._transform_fn(restored)
    assert result == {'params': {'a': 10}}

  def test_missing_keys_in_restored_filled(self):
    """Missing keys in restored are filled from state."""
    state = {'params': {'a': 1, 'b': 2}}
    restored = {'old_params': {'a': 10}}
    item = self._make_item(
        state,
        {'params': 'old_params'},
        ignore_model_keys=('**.b',),
    )
    result = item._transform_fn(restored)
    assert result == {'params': {'a': 10, 'b': 2}}

  def test_extra_keys_not_matched_raises(self):
    """Extra keys not matched by ignore_restored_keys raise an error."""
    state = {'params': {'a': 1}}
    restored = {'old_params': {'a': 10, 'b': 20}}
    item = self._make_item(
        state,
        {'params': 'old_params'},
        ignore_restored_keys=('**.c',),  # doesn't match 'b'
    )
    with pytest.raises(ValueError):
      item._transform_fn(restored)

  def test_missing_keys_not_matched_raises(self):
    """Missing keys not matched by ignore_model_keys raise an error."""
    state = {'params': {'a': 1, 'b': 2}}
    restored = {'old_params': {'a': 10}}
    item = self._make_item(
        state,
        {'params': 'old_params'},
        ignore_model_keys=('**.c',),  # doesn't match 'b'
    )
    with pytest.raises(ValueError):
      item._transform_fn(restored)

  def test_glob_double_star_extra(self):
    """**.bias removes bias at any nesting level."""
    state = {'params': {'layer': {'kernel': 1}}}
    restored = {'old': {'layer': {'kernel': 10, 'bias': 20}}}
    item = self._make_item(
        state,
        {'params': 'old'},
        ignore_restored_keys=('**.bias',),
    )
    result = item._transform_fn(restored)
    assert result == {'params': {'layer': {'kernel': 10}}}

  def test_glob_double_star_missing(self):
    """**.bias fills bias from state at any nesting level."""
    state = {'params': {'layer': {'kernel': 1, 'bias': 2}}}
    restored = {'old': {'layer': {'kernel': 10}}}
    item = self._make_item(
        state,
        {'params': 'old'},
        ignore_model_keys=('**.bias',),
    )
    result = item._transform_fn(restored)
    assert result == {'params': {'layer': {'kernel': 10, 'bias': 2}}}

  def test_both_extra_and_missing(self):
    """Handle both extra and missing keys simultaneously."""
    state = {'params': {'a': 1, 'c': 3}}
    restored = {'old': {'a': 10, 'b': 20}}
    item = self._make_item(
        state,
        {'params': 'old'},
        ignore_restored_keys=('**.b',),
        ignore_model_keys=('**.c',),
    )
    result = item._transform_fn(restored)
    assert result == {'params': {'a': 10, 'c': 3}}

  def test_no_ignore_keys_matching_structure(self):
    """Without ignore keys, matching structures work fine."""
    state = {'params': {'x': 1}}
    restored = {'params': {'x': 10}}
    item = self._make_item(state, {'params': 'params'})
    result = item._transform_fn(restored)
    assert result == {'params': {'x': 10}}

  def test_nested_extra_and_missing(self):
    """Extra and missing at nested levels with glob."""
    state = {'params': {'l1': {'a': 1}, 'l2': {'a': 2}}}
    restored = {'old': {'l1': {'a': 10, 'b': 11}, 'l2': {'a': 20, 'b': 21}}}
    item = self._make_item(
        state,
        {'params': 'old'},
        ignore_restored_keys=('**.b',),
    )
    result = item._transform_fn(restored)
    assert result == {'params': {'l1': {'a': 10}, 'l2': {'a': 20}}}


def _assert_not_all_close(*args):
  with pytest.raises(AssertionError):
    chex.assert_trees_all_close(*args)
