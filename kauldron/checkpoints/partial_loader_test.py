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

import contextlib
import pathlib
from typing import Any

import chex
import flax.struct
import jax
import jax.numpy as jnp
from kauldron import kd
from examples import mnist_autoencoder
from kauldron.testing import assert_utils
import numpy as np
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
  loader = kd.ckpts.PartialKauldronLoader(
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
      old_state.params['encoder'],  # pyrefly: ignore[unsupported-operation]
  )
  # Make sure weight were different
  _assert_not_all_close(
      init_state.params['decoder']['layers_1'],  # pyrefly: ignore[unsupported-operation]
      new_state.params['decoder']['layers_1'],
  )
  # Encoder is untouched, as well as layer 0
  chex.assert_trees_all_close(
      new_state.params['encoder'],
      init_state.params['encoder'],  # pyrefly: ignore[unsupported-operation]
  )
  chex.assert_trees_all_close(
      new_state.params['decoder']['layers_0'],
      init_state.params['decoder']['layers_0'],  # pyrefly: ignore[unsupported-operation]
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
      old_state.params['encoder'],  # pyrefly: ignore[unsupported-operation]
  )


def _make_trainer(
    tmp_path: pathlib.Path,
    name: str,
    decoder,
    seed: int,
) -> kd.train.Trainer:
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = str(tmp_path / name)
  cfg.model = kd_cfg.nn.FlatAutoencoder(
      inputs='batch.image',
      encoder=nn.Dense(features=_NUM_FEATURES),
      decoder=decoder,
  )
  cfg.train_ds.batch_size = 1
  cfg.seed = seed
  trainer = kd.konfig.resolve(cfg)
  with tfds.testing.mock_data():
    _ = trainer.train_ds.element_spec
  return trainer


@contextlib.contextmanager
def _forbid_host_transfers(monkeypatch: pytest.MonkeyPatch):
  """Simulates multi-host FSDP arrays by forbidding jax.Array -> np.asarray.

  Models `eval_fragments._coerce_literal` calling `np.asarray` on live
  `jax.Array` constants captured in `transform_fn` jaxprs.
  """
  orig_asarray = np.asarray

  def _forbid_jax_array_to_numpy(arr, *args, **kwargs):
    if isinstance(arr, jax.Array):
      raise RuntimeError(
          'Fetching value for jax.Array that spans non-addressable devices'
      )
    return orig_asarray(arr, *args, **kwargs)

  with monkeypatch.context() as m:
    m.setattr(np, 'asarray', _forbid_jax_array_to_numpy)
    yield


def _assert_trees_all_identical(tree_a, tree_b):
  leaves_a, treedef_a = jax.tree_util.tree_flatten(tree_a)
  leaves_b, treedef_b = jax.tree_util.tree_flatten(tree_b)
  assert treedef_a == treedef_b
  for leaf_a, leaf_b in zip(leaves_a, leaves_b, strict=True):
    assert leaf_a is leaf_b


def test_loader_ignore_model_keys_avoids_host_transfer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
  """ignore_model_keys must not pass live jax.Arrays into eval_fragments."""
  old_trainer_inst = _make_trainer(
      tmp_path,
      'old_workdir',
      nn.Sequential([nn.Dense(features=28 * 28)]),
      seed=1,
  )
  new_trainer_inst = _make_trainer(
      tmp_path,
      'new_workdir',
      nn.Sequential([
          nn.Dense(features=28 * 28),
          nn.Dense(features=28 * 28),
      ]),
      seed=0,
  )

  old_state = old_trainer_inst.init_state()
  old_trainer_inst.checkpointer.save(old_state, step=0)
  old_trainer_inst.checkpointer.wait_until_finished()
  init_state = new_trainer_inst.init_state()

  with _forbid_host_transfers(monkeypatch):
    loader = kd.ckpts.PartialKauldronLoader(
        workdir=old_trainer_inst.workdir,  # pylint: disable=missing-kwoa
        new_to_old={'params.decoder': 'params.decoder'},
        # Include both a present key (layers_0) and a missing key (layers_1) to
        # verify present keys are still restored while missing ones are kept.
        ignore_model_keys=('**.layers_0.**', '**.layers_1.**'),
    )
    new_state = loader.transform(init_state)

  assert_utils.assert_trees_all_same_type(init_state, new_state)
  assert old_state.params is not None
  assert init_state.params is not None
  assert new_state.params is not None
  chex.assert_trees_all_close(
      new_state.params['decoder']['layers_0'],
      old_state.params['decoder']['layers_0'],
  )
  chex.assert_trees_all_close(
      new_state.params['decoder']['layers_1'],
      init_state.params['decoder']['layers_1'],
  )
  _assert_trees_all_identical(
      new_state.params['decoder']['layers_1'],
      init_state.params['decoder']['layers_1'],
  )
  _assert_not_all_close(
      init_state.params['decoder']['layers_0'],
      new_state.params['decoder']['layers_0'],
  )


def test_loader_ignore_restored_keys_avoids_host_transfer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
  """ignore_restored_keys must not pass live jax.Arrays into eval_fragments."""
  decoder = nn.Sequential([
      nn.Dense(features=28 * 28),
      nn.Dense(features=28 * 28),
  ])
  old_trainer_inst = _make_trainer(tmp_path, 'old_workdir', decoder, seed=1)
  new_trainer_inst = _make_trainer(tmp_path, 'new_workdir', decoder, seed=0)

  old_state = old_trainer_inst.init_state()
  old_trainer_inst.checkpointer.save(old_state, step=0)
  old_trainer_inst.checkpointer.wait_until_finished()
  init_state = new_trainer_inst.init_state()

  with _forbid_host_transfers(monkeypatch):
    loader = kd.ckpts.PartialKauldronLoader(
        workdir=old_trainer_inst.workdir,  # pylint: disable=missing-kwoa
        new_to_old={'params.decoder': 'params.decoder'},
        ignore_restored_keys=('**.layers_1.**',),
    )
    new_state = loader.transform(init_state)

  assert_utils.assert_trees_all_same_type(init_state, new_state)
  assert old_state.params is not None
  assert init_state.params is not None
  assert new_state.params is not None
  chex.assert_trees_all_close(
      new_state.params['decoder']['layers_0'],
      old_state.params['decoder']['layers_0'],
  )
  chex.assert_trees_all_close(
      new_state.params['decoder']['layers_1'],
      init_state.params['decoder']['layers_1'],
  )
  _assert_trees_all_identical(
      new_state.params['decoder']['layers_1'],
      init_state.params['decoder']['layers_1'],
  )
  _assert_not_all_close(
      init_state.params['decoder']['layers_0'],
      new_state.params['decoder']['layers_0'],
  )
  _assert_not_all_close(
      old_state.params['decoder']['layers_1'],
      new_state.params['decoder']['layers_1'],
  )


@flax.struct.dataclass
class _DummyState(kd.ckpts.items.StandardCheckpointItem):
  params: Any
  opt_state: Any = None


def _save_dummy_state(
    workdir: pathlib.Path,
    state: _DummyState,
    *,
    step: int = 0,
) -> None:
  ckpt = kd.ckpts.Checkpointer(workdir=workdir, save_interval_steps=1)
  ckpt.save(state, step=step)
  ckpt.wait_until_finished()
  ckpt.close()


def _make_dummy_loader(workdir: pathlib.Path, **kwargs):
  return kd.ckpts.PartialKauldronLoader(workdir=workdir, **kwargs)  # pylint: disable=missing-kwoa


def test_forbid_host_transfers_catches_unplaceholdered_item(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
  """Canary verifying `_forbid_host_transfers` trips without placeholders."""
  workdir = tmp_path / 'ckpt'
  _save_dummy_state(workdir, _DummyState(params={'a': jnp.ones((2,))}))
  init_state = _DummyState(
      params={'a': jnp.zeros((2,)), 'b': jnp.full((2,), 3.0)}
  )
  item = kd.ckpts.partial_loader._PartialRestoreCheckpointItem(
      init_state,
      new_to_old={'params': 'params'},
      ignore_model_keys=('**.b',),
      use_placeholders=False,
  )
  ckpt = kd.ckpts.Checkpointer(workdir=workdir, save_interval_steps=1)
  try:
    with _forbid_host_transfers(monkeypatch):
      with pytest.raises(RuntimeError, match='non-addressable devices'):
        ckpt.restore(item)
  finally:
    ckpt.close()


def test_loader_missing_checkpoint_raises_file_not_found(
    tmp_path: pathlib.Path,
):
  init_state = _DummyState(params={'a': jnp.zeros((2,))})
  loader = _make_dummy_loader(
      tmp_path / 'empty_workdir',
      new_to_old={'params': 'params'},
      ignore_model_keys=('**.b',),
  )
  with pytest.raises(FileNotFoundError, match='No checkpoint found'):
    loader.transform(init_state)


def test_loader_unmatched_missing_key_raises(tmp_path: pathlib.Path):
  workdir = tmp_path / 'ckpt'
  _save_dummy_state(workdir, _DummyState(params={'a': jnp.ones((2,))}))
  init_state = _DummyState(
      params={'a': jnp.zeros((2,)), 'b': jnp.full((2,), 3.0)}
  )
  loader = _make_dummy_loader(
      workdir,
      new_to_old={'params': 'params'},
      ignore_model_keys=('**.other',),
  )
  with pytest.raises(ValueError, match='Missing key in restored checkpoint'):
    loader.transform(init_state)


def test_loader_all_leaves_ignored_still_checks_unexpected_keys(
    tmp_path: pathlib.Path,
):
  workdir = tmp_path / 'ckpt'
  _save_dummy_state(workdir, _DummyState(params={'old_only': jnp.ones((2, 2))}))
  init_state = _DummyState(params={'new_only': jnp.full((2, 2), 6.0)})
  loader = _make_dummy_loader(
      workdir,
      new_to_old={'params': 'params'},
      ignore_model_keys=('**.new_only',),
  )
  with pytest.raises(ValueError, match='Unexpected key in restored checkpoint'):
    loader.transform(init_state)


@pytest.mark.parametrize(
    (
        'old_state_fn',
        'init_state_fn',
        'new_to_old',
        'ignore_model_keys',
        'ignore_restored_keys',
        'expected_state_fn',
        'kept_paths',
    ),
    [
        pytest.param(
            lambda: _DummyState(
                params={
                    'empty_submod': {},
                    'layers_0': {'kernel': jnp.ones((2, 2))},
                }
            ),
            lambda: _DummyState(
                params={
                    'empty_submod': {},
                    'layers_0': {'kernel': jnp.zeros((2, 2))},
                    'layers_1': {'kernel': jnp.full((2, 2), 3.0)},
                }
            ),
            {'params': 'params'},
            ('**.layers_1.**',),
            (),
            lambda: _DummyState(
                params={
                    'empty_submod': {},
                    'layers_0': {'kernel': jnp.ones((2, 2))},
                    'layers_1': {'kernel': jnp.full((2, 2), 3.0)},
                }
            ),
            ('params.layers_1.kernel',),
            id='preexisting_empty_dict',
        ),
        pytest.param(
            lambda: _DummyState(
                params={
                    'my-layer': {'kernel': jnp.ones((2, 2))},
                    '0': {'bias': jnp.full((2,), 2.0)},
                }
            ),
            lambda: _DummyState(
                params={
                    'my-layer': {'kernel': jnp.zeros((2, 2))},
                    '0': {'bias': jnp.zeros((2,))},
                    'new-layer': {'kernel': jnp.full((2, 2), 5.0)},
                }
            ),
            {'params': 'params'},
            ('**.kernel', '**.bias'),
            (),
            lambda: _DummyState(
                params={
                    'my-layer': {'kernel': jnp.ones((2, 2))},
                    '0': {'bias': jnp.full((2,), 2.0)},
                    'new-layer': {'kernel': jnp.full((2, 2), 5.0)},
                }
            ),
            ("params['new-layer'].kernel",),
            id='non_identifier_dict_keys',
        ),
        pytest.param(
            lambda: _DummyState(
                params={
                    'blocks': [{'kernel': jnp.ones((2, 2))}],
                    'other': jnp.full((2,), 4.0),
                },
                opt_state=({'mu': {'a': jnp.ones((2,))}}, {}),
            ),
            lambda: _DummyState(
                params={
                    'blocks': [{
                        'kernel': jnp.zeros((2, 2)),
                        'bias': jnp.full((2,), 7.0),
                    }],
                    'other': jnp.zeros((2,)),
                },
                opt_state=(
                    {'mu': {'a': jnp.zeros((2,)), 'b': jnp.full((2,), 9.0)}},
                    {},
                ),
            ),
            {'params': 'params', 'opt_state': 'opt_state'},
            ('**.blocks', '**.bias', '**.b'),
            (),
            lambda: _DummyState(
                params={
                    'blocks': [{
                        'kernel': jnp.ones((2, 2)),
                        'bias': jnp.full((2,), 7.0),
                    }],
                    'other': jnp.full((2,), 4.0),
                },
                opt_state=(
                    {'mu': {'a': jnp.ones((2,)), 'b': jnp.full((2,), 9.0)}},
                    {},
                ),
            ),
            ('params.blocks[0].bias', 'opt_state[0].mu.b'),
            id='list_and_tuple_containers',
        ),
        pytest.param(
            lambda: _DummyState(
                params={
                    'encoder': {'kernel': jnp.ones((2, 2))},
                    'head_alt': {'kernel': jnp.full((2, 2), 8.0)},
                }
            ),
            lambda: _DummyState(
                params={
                    'encoder': {'kernel': jnp.zeros((2, 2))},
                    'head': {'kernel': jnp.full((2, 2), 3.0)},
                }
            ),
            {'params': 'params', 'params.head': 'params.head_alt'},
            ('**.head.**',),
            ('**.head_alt.**',),
            lambda: _DummyState(
                params={
                    'encoder': {'kernel': jnp.ones((2, 2))},
                    'head': {'kernel': jnp.full((2, 2), 8.0)},
                }
            ),
            (),
            id='overlapping_new_to_old_paths',
        ),
        pytest.param(
            lambda: _DummyState(params={'old_only': jnp.ones((2, 2))}),
            lambda: _DummyState(params={'new_only': jnp.full((2, 2), 6.0)}),
            {'params': 'params'},
            ('**.new_only',),
            ('**.old_only',),
            lambda: _DummyState(params={'new_only': jnp.full((2, 2), 6.0)}),
            ('params.new_only',),
            id='all_leaves_ignored',
        ),
        pytest.param(
            lambda: _DummyState(params={'scalar': jnp.array(1.0)}),
            lambda: _DummyState(params={'scalar': jnp.array(42.0)}),
            {'params.scalar': 'params.scalar'},
            (),
            ('**',),
            lambda: _DummyState(params={'scalar': jnp.array(42.0)}),
            ('params.scalar',),
            id='bare_leaf_ignored',
        ),
    ],
)
def test_loader_ignore_keys_edge_cases(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    old_state_fn,
    init_state_fn,
    new_to_old,
    ignore_model_keys,
    ignore_restored_keys,
    expected_state_fn,
    kept_paths,
):
  workdir = tmp_path / 'ckpt'
  _save_dummy_state(workdir, old_state_fn())
  init_state = init_state_fn()
  loader = _make_dummy_loader(
      workdir,
      new_to_old=new_to_old,
      ignore_model_keys=ignore_model_keys,
      ignore_restored_keys=ignore_restored_keys,
  )
  with _forbid_host_transfers(monkeypatch):
    new_state = loader.transform(init_state)
    new_state = loader.transform_after_optimizer(new_state)

  assert_utils.assert_trees_all_same_type(init_state, new_state)
  chex.assert_trees_all_close(new_state, expected_state_fn())
  for path in kept_paths:
    _assert_trees_all_identical(
        kd.kontext.get_by_path(new_state, path),
        kd.kontext.get_by_path(init_state, path),
    )


def test_loader_real_optax_opt_state_with_ignore_keys(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
  old_trainer_inst = _make_trainer(
      tmp_path,
      'old_workdir',
      nn.Sequential([nn.Dense(features=28 * 28)]),
      seed=1,
  )
  new_trainer_inst = _make_trainer(
      tmp_path,
      'new_workdir',
      nn.Sequential([
          nn.Dense(features=28 * 28),
          nn.Dense(features=28 * 28),
      ]),
      seed=0,
  )
  with tfds.testing.mock_data():
    old_state = old_trainer_inst.init_state()
    batch = next(iter(old_trainer_inst.train_ds))
    old_state, _ = old_trainer_inst.trainstep.step(old_state, batch)

  old_trainer_inst.checkpointer.save(old_state, step=0)
  old_trainer_inst.checkpointer.wait_until_finished()
  init_state = new_trainer_inst.init_state()

  loader = _make_dummy_loader(
      old_trainer_inst.workdir,
      new_to_old={'params.decoder': 'params.decoder', 'opt_state': 'opt_state'},
      ignore_model_keys=('**.layers_1.**',),
  )
  with _forbid_host_transfers(monkeypatch):
    new_state = loader.transform(init_state)
    new_state = loader.transform_after_optimizer(new_state)

  assert_utils.assert_trees_all_same_type(init_state, new_state)
  assert old_state.params is not None
  assert init_state.params is not None
  assert new_state.params is not None
  assert old_state.opt_state is not None
  assert init_state.opt_state is not None
  assert new_state.opt_state is not None
  chex.assert_trees_all_close(
      new_state.params['decoder']['layers_0'],
      old_state.params['decoder']['layers_0'],
  )
  _assert_trees_all_identical(
      new_state.params['decoder']['layers_1'],
      init_state.params['decoder']['layers_1'],
  )
  chex.assert_trees_all_close(
      new_state.opt_state[0].mu['decoder']['layers_0'],
      old_state.opt_state[0].mu['decoder']['layers_0'],
  )
  _assert_not_all_close(
      new_state.opt_state[0].mu['decoder']['layers_0'],
      init_state.opt_state[0].mu['decoder']['layers_0'],
  )
  _assert_trees_all_identical(
      new_state.opt_state[0].mu['decoder']['layers_1'],
      init_state.opt_state[0].mu['decoder']['layers_1'],
  )


def test_loader_explicit_step_with_ignore_keys(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
  workdir = tmp_path / 'ckpt'
  _save_dummy_state(
      workdir, _DummyState(params={'a': jnp.ones((2,))}), step=0
  )
  _save_dummy_state(
      workdir,
      _DummyState(params={'a': jnp.full((2,), 2.0), 'b': jnp.full((2,), 3.0)}),
      step=1,
  )
  init_state = _DummyState(
      params={'a': jnp.zeros((2,)), 'b': jnp.full((2,), 9.0)}
  )
  loader = _make_dummy_loader(
      workdir,
      step=0,
      new_to_old={'params': 'params'},
      ignore_model_keys=('**.b',),
  )
  with _forbid_host_transfers(monkeypatch):
    new_state = loader.transform(init_state)

  chex.assert_trees_all_close(new_state.params['a'], jnp.ones((2,)))
  _assert_trees_all_identical(new_state.params['b'], init_state.params['b'])


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

  def test_use_placeholders_returns_placeholder_for_ignored_keys(self):
    state = {'params': {'a': 1, 'b': 2, 'c': 3}}
    restored = {'old': {'a': 10, 'b': 20}}
    item = kd.ckpts.partial_loader._PartialRestoreCheckpointItem(
        state=state,
        new_to_old={'params': 'old'},
        ignore_restored_keys=('**.b',),
        ignore_model_keys=('**.c',),
        use_placeholders=True,
    )
    result = item._transform_fn(restored)
    assert result == {'params': {'a': 10, 'b': ..., 'c': ...}}


def _assert_not_all_close(*args):
  with pytest.raises(AssertionError):
    chex.assert_trees_all_close(*args)


@pytest.fixture
def matching_trainer(tmp_path: pathlib.Path) -> kd.train.Trainer:
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = str(tmp_path / 'matching_workdir')

  cfg.model = kd_cfg.nn.FlatAutoencoder(
      inputs='batch.image',
      encoder=nn.Dense(features=_NUM_FEATURES),
      decoder=nn.Dense(features=28 * 28),
  )

  cfg.train_ds.batch_size = 1
  cfg.seed = 2
  cfg = kd.konfig.resolve(cfg)
  with tfds.testing.mock_data():
    _ = cfg.train_ds.element_spec
  return cfg


def test_simple_checkpoint_loader(
    matching_trainer: kd.train.Trainer,
    old_trainer: kd.train.Trainer,
):  # pylint: disable=redefined-outer-name
  # 1. Initialize old trainer, run 1 step to update params and opt_state,
  # and save a checkpoint
  with tfds.testing.mock_data():
    old_state = old_trainer.init_state()
    batch = next(iter(old_trainer.train_ds))
    old_state, _ = old_trainer.trainstep.step(old_state, batch)

  old_trainer.checkpointer.save(old_state, step=0)
  old_trainer.checkpointer.wait_until_finished()

  # 2. Initialize state from matching_trainer
  init_state = matching_trainer.init_state()

  # Make sure initial weights and opt_state are different
  _assert_not_all_close(init_state.params, old_state.params)
  _assert_not_all_close(init_state.opt_state, old_state.opt_state)

  # 3. Apply transform with load_opt_state=False
  loader = kd.ckpts.SimpleCheckpointLoader(
      workdir=old_trainer.workdir,
      load_opt_state=False,
  )
  new_state = loader.transform(init_state)

  # params and collections should be successfully loaded and equal to old_state
  chex.assert_trees_all_close(new_state.params, old_state.params)
  chex.assert_trees_all_close(new_state.collections, old_state.collections)

  # opt_state should remain untouched (equal to matching_trainer's init_state)
  chex.assert_trees_all_close(new_state.opt_state, init_state.opt_state)
  _assert_not_all_close(new_state.opt_state, old_state.opt_state)

  # 4. Apply transform_after_optimizer with load_opt_state=True
  loader_with_opt = kd.ckpts.SimpleCheckpointLoader(
      workdir=old_trainer.workdir,
      load_opt_state=True,
  )
  new_state_with_opt = loader_with_opt.transform_after_optimizer(new_state)

  # opt_state should also be successfully loaded and equal to old_state's
  # opt_state
  chex.assert_trees_all_close(new_state_with_opt.opt_state, old_state.opt_state)
  # Verify that params and collections remain correctly loaded after
  # opt_state restoration
  chex.assert_trees_all_close(new_state_with_opt.params, old_state.params)
  chex.assert_trees_all_close(
      new_state_with_opt.collections, old_state.collections
  )
