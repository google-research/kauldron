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

"""End-to-end tests for Kauldron training, checkpointing, and evaluation."""

import functools
import os
from unittest import mock

from etils import epath
from flax import linen as nn
import jax
from kauldron import kd
from kauldron.evals import eval_impl
from examples import mnist_autoencoder
from kauldron.testing import assert_utils
from kauldron.train import metric_writer
from kauldron.train import train_step
from kauldron.utils import constants
import numpy as np
import pytest
import tensorflow_datasets as tfds


def _make_core_loop_cfg(
    workdir: epath.Path,
    *,
    num_train_steps: int = 4,
):
  """Builds a small CPU-friendly Trainer config covering the core loop."""
  cfg = mnist_autoencoder.get_config()
  cfg.seed = 2712828
  cfg.workdir = os.fspath(workdir)
  cfg.num_train_steps = num_train_steps
  cfg.log_metrics_every = 1
  cfg.log_summaries_every = 2

  # Disable PyGrain multiprocessing to keep CPU tests fast and deterministic.
  cfg.train_ds.batch_size = 4
  cfg.train_ds.num_workers = 0
  cfg.eval_ds = cfg.evals.eval.ds
  cfg.eval_ds.batch_size = 2
  cfg.eval_ds.num_workers = 0

  with kd.konfig.mock_modules():
    cfg.model.encoder = nn.Sequential([
        nn.Dense(features=4),
        nn.BatchNorm(use_running_average=False),
    ])
    cfg.checkpointer = kd.ckpts.Checkpointer(
        save_interval_steps=2,
        max_to_keep=3,
    )
    cfg.evals = {
        'in_loop': kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(2),
            num_batches=2,
        ),
        'final_eval': kd.evals.Evaluator(
            run=kd.evals.Final(),
            num_batches=2,
        ),
        'standalone_every': kd.evals.Evaluator(
            run=kd.evals.StandaloneEveryCheckpoint(),
            num_batches=2,
        ),
        'standalone_last': kd.evals.Evaluator(
            run=kd.evals.StandaloneLastCheckpoint(),
            num_batches=2,
        ),
    }
  return cfg


def test_end2end(tmp_path: epath.Path):
  """Runs a single training step with BatchNorm state updates."""
  # Load config and reduce size
  cfg = mnist_autoencoder.get_config()
  cfg.train_ds.batch_size = 2
  cfg.train_ds.num_workers = 0
  cfg.evals.eval.ds.batch_size = 2
  cfg.evals.eval.ds.num_workers = 0
  with kd.konfig.mock_modules():
    cfg.model.encoder = nn.Sequential([
        nn.Dense(features=3),
        nn.BatchNorm(use_running_average=False),
    ])
  cfg.num_train_steps = 1
  cfg.workdir = os.fspath(tmp_path)

  trainer = kd.konfig.resolve(cfg)

  # Launch train
  with trainer.checkpointer:
    with tfds.testing.mock_data(num_examples=10):
      state, _ = trainer.train()

    assert list(state.collections) == ['batch_stats']


class _SimulatedPreemptionError(Exception):
  """Sentinel raised mid-loop to simulate worker preemption."""


def test_core_loop_checkpoint_preempt_resume_and_eval(tmp_path: epath.Path):
  """Verifies checkpointing, preemption/resume state equivalence, and evals."""
  resumed_workdir = tmp_path / 'resumed_run'
  ref_workdir = tmp_path / 'reference_run'
  num_train_steps = 4

  with tfds.testing.mock_data(num_examples=24):
    # 1. Train part-way (steps 0, 1, 2 -> saves ckpt 0 and ckpt 2, plus
    # uncommitted step 2 update) and simulate mid-loop preemption at `i == 3`
    # so post-loop `final_eval` and `train_complete.txt` do not run.
    preempted_trainer = kd.konfig.resolve(
        _make_core_loop_cfg(resumed_workdir, num_train_steps=num_train_steps)
    )
    with preempted_trainer.checkpointer:
      init_state = preempted_trainer.init_state()
      orig_step = train_step.TrainStep.step
      preempted_state = None
      completed_steps = 0

      def _step_until_preempt(self, state, *args, **kwargs):
        nonlocal preempted_state, completed_steps
        if completed_steps == 3:
          with jax.transfer_guard('allow'):
            preempted_trainer.checkpointer.wait_until_finished()
          preempted_state = state
          raise _SimulatedPreemptionError()
        state, aux = orig_step(self, state, *args, **kwargs)
        completed_steps += 1
        preempted_state = state
        return state, aux

      with (
          mock.patch.object(train_step.TrainStep, 'step', _step_until_preempt),
          pytest.raises(_SimulatedPreemptionError),
      ):
        preempted_trainer.train()

      preempted_trainer.writer.wait_for_background_flush()
      for ev in preempted_trainer.evals.values():
        ev.writer.wait_for_background_flush()

      complete_marker = resumed_workdir / eval_impl.TRAIN_COMPLETE_FILENAME
      assert preempted_state is not None
      assert int(preempted_state.step) == 3
      assert list(preempted_trainer.checkpointer.all_steps) == [0, 2]
      assert preempted_trainer.checkpointer.latest_step == 2
      assert (resumed_workdir / constants.CONFIG_FILENAME).exists()
      assert (resumed_workdir / constants.ELEMENT_SPEC_FILENAME).exists()
      assert not (resumed_workdir / 'final_eval').exists()
      assert not complete_marker.exists()

    # 2. Resume in a fresh Trainer from ckpt 2 and finish training (steps 2..4).
    resumed_trainer = kd.konfig.resolve(
        _make_core_loop_cfg(resumed_workdir, num_train_steps=num_train_steps)
    )
    with resumed_trainer.checkpointer:
      assert resumed_trainer.checkpointer.latest_step == 2

      resumed_state, resumed_aux = resumed_trainer.train()

      # Loop runs steps 0..num_train_steps inclusive, leaving step at N + 1.
      assert int(resumed_state.step) == num_train_steps + 1
      assert list(resumed_trainer.checkpointer.all_steps) == [0, 2, 4]
      assert resumed_trainer.checkpointer.latest_step == 4
      assert complete_marker.exists()
      assert (resumed_workdir / 'train').exists()
      assert (resumed_workdir / 'in_loop').exists()
      assert (resumed_workdir / 'final_eval').exists()

      train_out = resumed_aux.compute()
      assert np.isfinite(train_out.loss_values['losses/recon'])
      assert np.isfinite(train_out.metric_values['metrics/latent_norm'])

    # 3. Verify resumed state (params + batch_stats + opt_state + step) matches
    # an uninterrupted run, proving model state, optimizer state, and PyGrain
    # dataset iterator position all restored cleanly across preemption.
    ref_trainer = kd.konfig.resolve(
        _make_core_loop_cfg(ref_workdir, num_train_steps=num_train_steps)
    )
    with ref_trainer.checkpointer:
      ref_state, _ = ref_trainer.train()

    assert int(resumed_state.step) == int(ref_state.step)
    # 1e-5 absorbs minor XLA float32 reassociation drift across upgrades while
    # still catching missed/replayed batches or un-restored state (>1e-2).
    jax.tree.map(
        functools.partial(np.testing.assert_allclose, rtol=1e-5, atol=1e-5),
        resumed_state.params,
        ref_state.params,
    )
    jax.tree.map(
        functools.partial(np.testing.assert_allclose, rtol=1e-5, atol=1e-5),
        resumed_state.collections,
        ref_state.collections,
    )
    jax.tree.map(
        functools.partial(np.testing.assert_allclose, rtol=1e-5, atol=1e-5),
        resumed_state.opt_state,
        ref_state.opt_state,
    )
    # Confirm training updated both encoder and decoder weights away from init.
    assert not np.allclose(
        init_state.params['encoder']['layers_0']['kernel'],
        resumed_state.params['encoder']['layers_0']['kernel'],
    )
    assert not np.allclose(
        init_state.params['decoder']['kernel'],
        resumed_state.params['decoder']['kernel'],
    )


def test_continuous_eval_over_saved_checkpoints(tmp_path: epath.Path):
  """Verifies `continuous_eval` runs every- and last-checkpoint evaluators."""
  num_train_steps = 4
  with tfds.testing.mock_data(num_examples=24):
    train_trainer = kd.konfig.resolve(
        _make_core_loop_cfg(tmp_path, num_train_steps=num_train_steps)
    )
    with train_trainer.checkpointer:
      train_trainer.train()

    # Mock `iter_new_checkpoints` to avoid polling sleeps; live checkpoint
    # directory discovery/polling is not exercised in this unit test.
    assert not (tmp_path / 'standalone_every').exists()
    assert not (tmp_path / 'standalone_last').exists()
    eval_trainer = kd.konfig.resolve(
        _make_core_loop_cfg(tmp_path, num_train_steps=num_train_steps)
    )
    with eval_trainer.checkpointer:
      assert list(eval_trainer.checkpointer.all_steps) == [0, 2, 4]
      every_writer = eval_trainer.evals['standalone_every'].writer
      last_writer = eval_trainer.evals['standalone_last'].writer
      orig_write_scalars = metric_writer.KDMetricWriter.write_scalars
      every_logged: list[tuple[int, dict[str, float]]] = []
      last_logged: list[tuple[int, dict[str, float]]] = []

      def _iter_all_saved_checkpoints(self, *args, **kwargs):
        del args, kwargs
        self.reload()
        return iter(sorted(self.all_steps))

      def _spy_write_scalars(self, step, scalars):
        if self is every_writer:
          every_logged.append((int(step), dict(scalars)))
        elif self is last_writer:
          last_logged.append((int(step), dict(scalars)))
        return orig_write_scalars(self, step=step, scalars=scalars)

      with (
          mock.patch.object(
              kd.ckpts.Checkpointer,
              'iter_new_checkpoints',
              autospec=True,
              side_effect=_iter_all_saved_checkpoints,
          ),
          mock.patch.object(
              metric_writer.KDMetricWriter,
              'write_scalars',
              autospec=True,
              side_effect=_spy_write_scalars,
          ),
      ):
        standalone_aux = eval_trainer.continuous_eval(
            ['standalone_every', 'standalone_last']
        )

      assert list(eval_trainer.checkpointer.all_steps) == [0, 2, 4]
      assert [step for step, _ in every_logged] == [0, 2, 4]
      assert [step for step, _ in last_logged] == [4]
      assert not np.allclose(
          every_logged[0][1]['losses/recon'],
          every_logged[2][1]['losses/recon'],
      )
      np.testing.assert_allclose(
          every_logged[2][1]['losses/recon'],
          last_logged[0][1]['losses/recon'],
          rtol=1e-6,
          atol=1e-6,
      )
      assert set(standalone_aux.keys()) == {
          'standalone_every',
          'standalone_last',
      }
      every_out = standalone_aux['standalone_every'].compute()
      last_out = standalone_aux['standalone_last'].compute()
      assert np.isfinite(every_out.loss_values['losses/recon'])
      assert np.isfinite(every_out.loss_values['losses/total'])
      assert np.isfinite(every_out.metric_values['metrics/latent_norm'])
      assert np.isfinite(last_out.loss_values['losses/recon'])
      assert np.isfinite(last_out.loss_values['losses/total'])
      assert np.isfinite(last_out.metric_values['metrics/latent_norm'])
      np.testing.assert_allclose(
          every_out.loss_values['losses/recon'],
          last_out.loss_values['losses/recon'],
          rtol=1e-6,
          atol=1e-6,
      )
      assert (tmp_path / 'standalone_every').exists()
      assert (tmp_path / 'standalone_last').exists()


def test_train_loop_shards_batches_across_devices(tmp_path: epath.Path):
  """Verifies `train_loop.train_impl` shards train batches before `step`."""
  assert_utils.require_two_devices()
  cfg = _make_core_loop_cfg(tmp_path, num_train_steps=0)
  with kd.konfig.mock_modules():
    cfg.checkpointer = kd.ckpts.NoopCheckpointer()
    cfg.evals = {}
  trainer = kd.konfig.resolve(cfg)

  shard_shapes: list[tuple[tuple[int, ...], list[tuple[int, ...]]]] = []
  orig_step = train_step.TrainStep.step

  def _spy_step(self, state, batch, *args, **kwargs):
    img = batch['image']
    shard_shapes.append(
        (img.shape, [s.data.shape for s in img.addressable_shards])
    )
    return orig_step(self, state, batch, *args, **kwargs)

  with (
      tfds.testing.mock_data(num_examples=8),
      trainer.checkpointer,
      mock.patch.object(train_step.TrainStep, 'step', _spy_step),
  ):
    trainer.train()

  assert shard_shapes == [((4, 28, 28, 1), [(2, 28, 28, 1), (2, 28, 28, 1)])]


def test_multi_device_sharding_end2end(tmp_path: epath.Path):
  """Verifies 2-device FSDPSharding training and checkpoint restoration."""
  assert_utils.require_two_devices()
  cfg = _make_core_loop_cfg(tmp_path, num_train_steps=2)
  with kd.konfig.mock_modules():
    cfg.sharding = kd.sharding.ShardingStrategy(
        batch=kd.sharding.FIRST_DIM,
        params=kd.sharding.FSDPSharding(min_size_to_shard_mb=0),
    )
    cfg.checkpointer = kd.ckpts.Checkpointer(save_interval_steps=2)
    cfg.evals = {
        'in_loop': kd.evals.Evaluator(
            run=kd.evals.EveryNSteps(2),
            num_batches=2,
        ),
    }

  trainer = kd.konfig.resolve(cfg)

  with tfds.testing.mock_data(num_examples=16):
    with trainer.checkpointer:
      state, _ = trainer.train()

      # FSDPSharding(min_size_to_shard_mb=0) shards along the largest dimension
      # divisible by the 2 devices: axis 0 for encoder (784, 4) -> (392, 4) and
      # axis 1 for decoder (4, 784) -> (4, 392).
      enc_kernel = state.params['encoder']['layers_0']['kernel']
      dec_kernel = state.params['decoder']['kernel']
      assert enc_kernel.shape == (784, 4)
      assert len(enc_kernel.addressable_shards) == 2
      assert enc_kernel.addressable_shards[0].data.shape == (392, 4)
      assert dec_kernel.shape == (4, 784)
      assert len(dec_kernel.addressable_shards) == 2
      assert dec_kernel.addressable_shards[0].data.shape == (4, 392)

    # Restore into a fresh 2-device trainer and verify sharding is preserved.
    restored_trainer = kd.konfig.resolve(cfg)
    with restored_trainer.checkpointer:
      restored_target = restored_trainer.init_state()
      restored_state = restored_trainer.checkpointer.restore(
          restored_target, step=2, donate=False
      )
      restored_enc_kernel = restored_state.params['encoder']['layers_0'][
          'kernel'
      ]
      restored_dec_kernel = restored_state.params['decoder']['kernel']
      assert int(restored_state.step) == 2
      assert restored_enc_kernel.sharding == enc_kernel.sharding
      assert restored_enc_kernel.addressable_shards[0].data.shape == (392, 4)
      assert restored_dec_kernel.sharding == dec_kernel.sharding
      assert restored_dec_kernel.addressable_shards[0].data.shape == (4, 392)
      assert not np.allclose(
          restored_target.params['encoder']['layers_0']['kernel'],
          restored_enc_kernel,
      )
      assert not np.allclose(
          restored_target.params['decoder']['kernel'],
          restored_dec_kernel,
      )
