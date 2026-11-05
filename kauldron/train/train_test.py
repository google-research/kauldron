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

    # 3. Verify resumed state (params + batch_stats + step) matches an
    # uninterrupted run, proving model state, optimizer state, and PyGrain
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
    # Confirm training updated both encoder and decoder weights away from init.
    assert not np.allclose(
        init_state.params['encoder']['layers_0']['kernel'],
        resumed_state.params['encoder']['layers_0']['kernel'],
    )
    assert not np.allclose(
        init_state.params['decoder']['kernel'],
        resumed_state.params['decoder']['kernel'],
    )

    # 4. Run `continuous_eval` (`StandaloneEveryCheckpoint` +
    # `StandaloneLastCheckpoint`) against saved checkpoints (0, 2, 4).
    # Mock `iter_new_checkpoints` to avoid polling sleeps; live checkpoint
    # directory discovery/polling is not exercised in this unit test.
    assert not (resumed_workdir / 'standalone_every').exists()
    assert not (resumed_workdir / 'standalone_last').exists()
    eval_trainer = kd.konfig.resolve(
        _make_core_loop_cfg(resumed_workdir, num_train_steps=num_train_steps)
    )
    with eval_trainer.checkpointer:
      every_writer = eval_trainer.evals['standalone_every'].writer
      last_writer = eval_trainer.evals['standalone_last'].writer
      orig_write_scalars = metric_writer.KDMetricWriter.write_scalars
      every_logged: list[tuple[int, dict[str, float]]] = []
      last_logged: list[tuple[int, dict[str, float]]] = []

      def _iter_all_saved_checkpoints(self, *args, **kwargs):
        del args, kwargs
        self.reload()
        assert list(self.all_steps) == [0, 2, 4]
        return iter([0, 2, 4])

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
      assert (resumed_workdir / 'standalone_every').exists()
      assert (resumed_workdir / 'standalone_last').exists()


def _require_two_cpu_devices() -> None:
  device_count = jax.local_device_count()
  if device_count < 2:
    if (
        'xla_force_host_platform_device_count=2' in os.environ.get(
            'XLA_FLAGS', ''
        )
        or os.environ.get('JAX_NUM_CPU_DEVICES') == '2'
    ):
      pytest.fail(
          'Expected >=2 CPU devices via XLA_FLAGS or JAX_NUM_CPU_DEVICES=2,'
          f' but JAX initialized with {device_count}.'
      )
    pytest.skip('Requires >=2 host CPU devices')


def test_eval_dataset_batches_sharded_across_devices(tmp_path: epath.Path):
  _require_two_cpu_devices()
  cfg = _make_core_loop_cfg(tmp_path, num_train_steps=1)
  trainer = kd.konfig.resolve(cfg)

  with trainer.checkpointer:
    with tfds.testing.mock_data(num_examples=8):
      eval_batch = next(iter(trainer.evals['in_loop'].ds_iter))

    assert eval_batch['image'].shape == (2, 28, 28, 1)
    assert len(eval_batch['image'].addressable_shards) == 2
    assert eval_batch['image'].addressable_shards[0].data.shape == (
        1,
        28,
        28,
        1,
    )


def test_multi_device_sharding_end2end(tmp_path: epath.Path):
  _require_two_cpu_devices()
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
          restored_target, step=2
      )
      restored_enc_kernel = restored_state.params['encoder']['layers_0'][
          'kernel'
      ]
      restored_dec_kernel = restored_state.params['decoder']['kernel']
      assert restored_enc_kernel.sharding == enc_kernel.sharding
      assert restored_enc_kernel.addressable_shards[0].data.shape == (392, 4)
      assert restored_dec_kernel.sharding == dec_kernel.sharding
      assert restored_dec_kernel.addressable_shards[0].data.shape == (4, 392)
