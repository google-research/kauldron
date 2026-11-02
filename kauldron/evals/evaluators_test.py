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

"""Tests for Kauldron evaluators."""

import os
import pathlib

from etils import epath
from flax import linen as nn
from kauldron import kd
from examples import mnist_autoencoder
from kauldron.testing import assert_utils
import numpy as np
import pytest
import tensorflow_datasets as tfds


def test_multi(tmp_path: pathlib.Path):
  """Resolves multiple evaluators and assigns each its dict key as `.name`."""
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = os.fspath(tmp_path)
  cfg.eval_ds = cfg.train_ds

  with kd.konfig.mock_modules():
    cfg.evals = {
        'test_eval': kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=1,
        ),
        'eval002': kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=1,
        ),
    }

  cfg = kd.konfig.resolve(cfg)

  assert cfg.evals['test_eval'].name == 'test_eval'
  assert cfg.evals['eval002'].name == 'eval002'


def _make_eval_cfg(tmp_path: epath.Path):
  """Builds a minimal CPU MNIST autoencoder Trainer config for eval tests."""
  # Starts from `mnist_autoencoder.get_config()` (`FlatAutoencoder` with L2
  # reconstruction loss `recon`, `latent_norm` metric, and Adam optimizer) and
  # shrinks the encoder (`features=4`), batch sizes (`train=4`, `eval=2`), and
  # PyGrain workers (`num_workers=0`) with `NoopCheckpointer` for fast CPU runs.
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = os.fspath(tmp_path)
  cfg.train_ds.batch_size = 4
  cfg.train_ds.num_workers = 0
  cfg.eval_ds = cfg.evals.eval.ds
  cfg.eval_ds.batch_size = 2
  cfg.eval_ds.num_workers = 0
  cfg.model.encoder.features = 4
  with kd.konfig.mock_modules():
    cfg.checkpointer = kd.ckpts.NoopCheckpointer()
  return cfg


def test_evaluator_every_n_steps_schedule(tmp_path: epath.Path):
  """Verifies `EveryNSteps(2, skip_first=True)` runs only at step 2."""
  cfg = _make_eval_cfg(tmp_path)
  with kd.konfig.mock_modules():
    cfg.evals = {
        'every_two': kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(2, skip_first=True),
            num_batches=2,
        ),
    }

  trainer = kd.konfig.resolve(cfg)

  with tfds.testing.mock_data(num_examples=12):
    state = trainer.init_state()
    res_step0 = trainer.evals['every_two'].maybe_eval(step=0, state=state)
    res_step1 = trainer.evals['every_two'].maybe_eval(step=1, state=state)
    aux_step2 = trainer.evals['every_two'].maybe_eval(step=2, state=state)

  assert res_step0 is None
  assert res_step1 is None
  assert aux_step2 is not None
  out_step2 = aux_step2.compute()
  assert np.isfinite(out_step2.loss_values['losses/recon'])
  assert np.isfinite(out_step2.loss_values['losses/total'])
  assert np.isfinite(out_step2.metric_values['metrics/latent_norm'])
  assert (tmp_path / 'every_two').exists()


def test_evaluator_once_schedule(tmp_path: epath.Path):
  """Verifies `Once(step=1)` evaluates at step 1 and skips steps 0 and 2."""
  cfg = _make_eval_cfg(tmp_path)
  with kd.konfig.mock_modules():
    cfg.evals = {
        'once_at_one': kd.evals.Evaluator(
            run=kd.evals.Once(step=1),
            num_batches=1,
        ),
    }

  trainer = kd.konfig.resolve(cfg)

  with tfds.testing.mock_data(num_examples=12):
    state = trainer.init_state()
    res_step0 = trainer.evals['once_at_one'].maybe_eval(step=0, state=state)
    aux_once = trainer.evals['once_at_one'].maybe_eval(step=1, state=state)
    res_step2 = trainer.evals['once_at_one'].maybe_eval(step=2, state=state)

  assert res_step0 is None
  assert aux_once is not None
  assert res_step2 is None
  assert np.isfinite(aux_once.compute().loss_values['losses/recon'])


def test_evaluator_standalone_schedule(tmp_path: epath.Path):
  """Verifies `StandaloneEveryCheckpoint` skips in-loop `maybe_eval` calls."""
  cfg = _make_eval_cfg(tmp_path)
  with kd.konfig.mock_modules():
    cfg.evals = {
        'standalone': kd.evals.Evaluator(
            run=kd.evals.StandaloneEveryCheckpoint(),
            num_batches=1,
        ),
    }

  trainer = kd.konfig.resolve(cfg)

  with tfds.testing.mock_data(num_examples=12):
    state = trainer.init_state()
    res_step0 = trainer.evals['standalone'].maybe_eval(step=0, state=state)
    res_step1 = trainer.evals['standalone'].maybe_eval(step=1, state=state)
    res_step2 = trainer.evals['standalone'].maybe_eval(step=2, state=state)

  assert res_step0 is None
  assert res_step1 is None
  assert res_step2 is None
  assert not (tmp_path / 'standalone').exists()


def test_evaluator_final_schedule(tmp_path: epath.Path):
  """Verifies `Final` skips in-loop `maybe_eval` and runs after `train()`."""
  cfg = _make_eval_cfg(tmp_path)
  cfg.num_train_steps = 1
  with kd.konfig.mock_modules():
    cfg.evals = {
        'final': kd.evals.Evaluator(
            run=kd.evals.Final(),
            num_batches=1,
        ),
    }

  trainer = kd.konfig.resolve(cfg)

  with tfds.testing.mock_data(num_examples=12):
    state = trainer.init_state()
    res_step0 = trainer.evals['final'].maybe_eval(step=0, state=state)
    res_step1 = trainer.evals['final'].maybe_eval(step=1, state=state)
    assert res_step0 is None
    assert res_step1 is None
    assert not (tmp_path / 'final').exists()

    trainer.train()

  assert (tmp_path / 'final').exists()


def test_evaluator_direct_evaluate_writes_metrics(tmp_path: epath.Path):
  """Verifies `evaluate()` returns finite metrics and creates its workdir."""
  cfg = _make_eval_cfg(tmp_path)
  with kd.konfig.mock_modules():
    cfg.evals = {
        'direct': kd.evals.Evaluator(
            run=kd.evals.StandaloneEveryCheckpoint(),
            num_batches=1,
        ),
    }

  trainer = kd.konfig.resolve(cfg)

  with tfds.testing.mock_data(num_examples=12):
    state = trainer.init_state()
    direct_aux = trainer.evals['direct'].evaluate(state=state, step=2)

  out = direct_aux.compute()
  assert np.isfinite(out.loss_values['losses/recon'])
  assert np.isfinite(out.metric_values['metrics/latent_norm'])
  assert (tmp_path / 'direct').exists()


def test_evaluator_sets_is_training_false_for_dropout(tmp_path: epath.Path):
  """Verifies `evaluate()` runs `kd.nn.Dropout` with `is_training=False`."""
  cfg = _make_eval_cfg(tmp_path)
  with kd.konfig.mock_modules():
    # Enable the `'dropout'` RNG stream during eval too so stochastic dropout
    # would fire if `Evaluator` did not pass `is_training=False`.
    cfg.rng_streams = kd.train.RngStreams([
        kd.train.RngStream('dropout', eval=True),
    ])
    cfg.model.encoder = nn.Sequential([
        nn.Dense(features=4),
        kd.nn.Dropout(rate=0.5),
    ])
    cfg.evals = {
        'eval': kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=1,
        ),
    }

  trainer = kd.konfig.resolve(cfg)

  with tfds.testing.mock_data(num_examples=12):
    state = trainer.init_state()
    eval_aux = trainer.evals['eval'].evaluate(state=state, step=0)
    eval_batch = next(iter(trainer.evals['eval'].ds_iter))
    det_preds = trainer.model.apply(
        {'params': state.params},
        eval_batch['image'],
        is_training_property=False,
    )
    expected_eval_loss = np.mean(
        np.square(det_preds['image'] - eval_batch['image'])
    )
    # Run `trainstep.step` last since it donates `state` buffers.
    _, train_aux = trainer.trainstep.step(
        state, eval_batch, return_losses=True
    )

  train_loss = train_aux.compute().loss_values['losses/recon']
  eval_loss = eval_aux.compute().loss_values['losses/recon']
  assert np.isfinite(train_loss)
  assert np.isfinite(eval_loss)
  assert not np.allclose(train_loss, eval_loss)
  np.testing.assert_allclose(
      eval_loss,
      expected_eval_loss,
      rtol=1e-6,
      atol=1e-6,
  )


@pytest.mark.parametrize(
    ('eval_key', 'explicit_name', 'error_match'),
    [
        ('train', None, 'Evaluator cannot be named `train`'),
        (
            'invalid name',
            None,
            'Evaluator name should be a valid Python identifier',
        ),
        (
            'eval_key',
            'other_name',
            'Evaluator name provided should match',
        ),
    ],
)
def test_evaluator_invalid_names_raise_value_error(
    tmp_path: epath.Path,
    eval_key: str,
    explicit_name: str | None,
    error_match: str,
):
  """Verifies invalid or mismatched evaluator names raise `ValueError`."""
  cfg = _make_eval_cfg(tmp_path)
  eval_kwargs = {} if explicit_name is None else {'name': explicit_name}
  with kd.konfig.mock_modules():
    cfg.evals = {
        eval_key: kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=1,
            **eval_kwargs,
        ),
    }

  with pytest.raises(ValueError, match=error_match):
    kd.konfig.resolve(cfg)


def test_evaluator_shards_batches_across_two_devices(tmp_path: epath.Path):
  """Verifies `Evaluator.ds_iter` shards batches across 2 local devices."""
  assert_utils.require_two_devices()
  cfg = _make_eval_cfg(tmp_path)
  with kd.konfig.mock_modules():
    cfg.evals = {
        'sharded_eval': kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(1),
            num_batches=2,
        ),
    }

  trainer = kd.konfig.resolve(cfg)

  with tfds.testing.mock_data(num_examples=12):
    state = trainer.init_state()
    eval_batch = next(iter(trainer.evals['sharded_eval'].ds_iter))
    aux = trainer.evals['sharded_eval'].evaluate(state=state, step=0)

  assert eval_batch['image'].shape == (2, 28, 28, 1)
  assert len(eval_batch['image'].addressable_shards) == 2
  assert eval_batch['image'].addressable_shards[0].data.shape == (1, 28, 28, 1)
  assert np.isfinite(aux.compute().loss_values['losses/recon'])
