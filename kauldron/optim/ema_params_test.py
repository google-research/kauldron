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

"""Tests for ema_params and UseEmaParams."""

import pathlib
from typing import Any, cast

import chex
from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron import kd
import numpy as np
import optax
import pytest


class _LinearModel(nn.Module):
  x: kd.kontext.Key = 'batch.x'

  @nn.compact
  def __call__(self, x: jax.Array) -> jax.Array:
    return nn.Dense(features=2, use_bias=False)(x)


def _make_trainstep(
    *,
    optimizer: optax.GradientTransformation,
    init_transform: kd.ckpts.InitTransform = kd.ckpts.NoopTransform(),
    seed: int = 0,
) -> kd.train.TrainStep:
  return kd.train.TrainStep(
      model=_LinearModel(),
      optimizer=optimizer,
      rng_streams=kd.train.RngStreams(seed=seed),
      sharding=kd.sharding.ShardingStrategy(),
      init_transform=init_transform,
      aux=kd.train.Auxiliaries(
          losses={'loss': kd.losses.L2(preds='preds', targets='batch.y')},
          metrics={},
          summaries={},
      ),
  )


def _init_train_state(
    step_fn: kd.train.TrainStep,
    *,
    skip_transforms: bool = False,
    skip_optimizer: bool = False,
) -> kd.train.TrainState:
  elem_spec = {
      'x': jax.ShapeDtypeStruct((2, 3), jnp.float32),
      'y': jax.ShapeDtypeStruct((2, 2), jnp.float32),
  }
  return step_fn.init(
      elem_spec,
      skip_transforms=skip_transforms,
      skip_optimizer=skip_optimizer,
  )


def _make_train_state(
    *,
    params: Any = None,
    opt_state: Any = None,
) -> kd.train.TrainState:
  return kd.train.TrainState(
      step=0,
      params={'w': jnp.array([1.0, 2.0])} if params is None else params,
      opt_state={} if opt_state is None else opt_state,
      collections={},
  )


def _save_two_step_ema_checkpoint(
    workdir: pathlib.Path,
    *,
    optimizer: optax.GradientTransformation,
) -> kd.train.TrainState:
  batch = {
      'x': jnp.ones((2, 3), dtype=jnp.float32),
      'y': jnp.zeros((2, 2), dtype=jnp.float32),
  }
  step_fn = _make_trainstep(optimizer=optimizer, seed=0)
  state = _init_train_state(step_fn)
  state, _ = step_fn.step(state, batch)
  state, _ = step_fn.step(state, batch)

  with kd.ckpts.Checkpointer(workdir=workdir, save_interval_steps=1) as ckpt:
    ckpt.save(state, step=int(state.step))
  return state


def test_ema_params_init_copies_initial_params_with_zero_count():
  tx: Any = kd.optim.ema_params(decay=0.5, debias=False)
  initial_params = {'w': jnp.array([2.0, 4.0], dtype=jnp.float32)}

  state = tx.init(initial_params)

  np.testing.assert_array_equal(state.count, jnp.array(0, jnp.int32))
  chex.assert_trees_all_close(state.ema_params, initial_params)


def test_ema_params_update_passes_updates_through_unchanged():
  tx: Any = kd.optim.ema_params(decay=0.5, debias=False)
  params = {'w': jnp.array([2.0, 4.0], dtype=jnp.float32)}
  updates = {'w': jnp.array([2.0, -2.0], dtype=jnp.float32)}
  state = tx.init(params)

  out_updates, _ = tx.update(updates, state, params)

  chex.assert_trees_all_close(out_updates, updates)


def test_ema_params_without_debias_averages_post_update_params():
  tx: Any = kd.optim.ema_params(decay=0.5, debias=False)
  params0 = {'w': jnp.array([2.0, 4.0], dtype=jnp.float32)}
  updates1 = {'w': jnp.array([2.0, -2.0], dtype=jnp.float32)}
  updates2 = {'w': jnp.array([4.0, 2.0], dtype=jnp.float32)}
  state0 = tx.init(params0)

  out_updates1, state1 = tx.update(updates1, state0, params0)
  params1 = optax.apply_updates(params0, out_updates1)
  _, state2 = tx.update(updates2, state1, params1)

  # Step 1: params1 = [4.0, 2.0],
  # ema1 = 0.5 * [2.0, 4.0] + 0.5 * [4.0, 2.0] = [3.0, 3.0]
  np.testing.assert_array_equal(state1.count, jnp.array(1, jnp.int32))
  chex.assert_trees_all_close(
      state1.ema_params,
      {'w': jnp.array([3.0, 3.0], dtype=jnp.float32)},
  )
  # Step 2: params2 = [8.0, 4.0],
  # ema2 = 0.5 * [3.0, 3.0] + 0.5 * [8.0, 4.0] = [5.5, 3.5]
  np.testing.assert_array_equal(state2.count, jnp.array(2, jnp.int32))
  chex.assert_trees_all_close(
      state2.ema_params,
      {'w': jnp.array([5.5, 3.5], dtype=jnp.float32)},
  )


def test_ema_params_with_debias_applies_warmup_decay_schedule():
  tx: Any = kd.optim.ema_params(decay=0.999, debias=True)
  params0 = {'w': jnp.array([1.0], dtype=jnp.float32)}
  updates1 = {'w': jnp.array([11.0], dtype=jnp.float32)}
  updates2 = {'w': jnp.array([2.0], dtype=jnp.float32)}
  state0 = tx.init(params0)

  out_updates1, state1 = tx.update(updates1, state0, params0)
  params1 = optax.apply_updates(params0, out_updates1)
  _, state2 = tx.update(updates2, state1, params1)

  # Step 1: count = 1, debiased_decay = min(0.999, 2 / 11) = 2 / 11.
  # new_params = [12.0], ema = (2/11)*1.0 + (9/11)*12.0 = 10.0.
  np.testing.assert_array_equal(state1.count, jnp.array(1, jnp.int32))
  chex.assert_trees_all_close(
      state1.ema_params, {'w': jnp.array([10.0], dtype=jnp.float32)}
  )
  # Step 2: count = 2, debiased_decay = min(0.999, 3 / 12) = 0.25.
  # new_params = [14.0], ema = 0.25*10.0 + 0.75*14.0 = 13.0.
  np.testing.assert_array_equal(state2.count, jnp.array(2, jnp.int32))
  chex.assert_trees_all_close(
      state2.ema_params, {'w': jnp.array([13.0], dtype=jnp.float32)}
  )


def test_ema_params_casts_state_to_accumulator_dtype():
  tx: Any = kd.optim.ema_params(
      decay=0.5, debias=False, accumulator_dtype=jnp.bfloat16
  )
  params = {'w': jnp.array([1.0, 2.0], dtype=jnp.float32)}
  updates = {'w': jnp.array([1.0, 1.0], dtype=jnp.float32)}

  init_state = tx.init(params)
  _, updated_state = tx.update(updates, init_state, params)

  assert init_state.ema_params['w'].dtype == jnp.bfloat16
  assert updated_state.ema_params['w'].dtype == jnp.bfloat16


def test_use_ema_params_replaces_params_from_unique_ema_state():
  ema_weights = {'w': jnp.array([10.0, 20.0])}
  state = _make_train_state(
      params={'w': jnp.array([1.0, 2.0])},
      opt_state={
          'adam': {'m': jnp.array([0.1, 0.2])},
          'ema': kd.optim.ema_params(decay=0.9).init(ema_weights),
      },
  )

  transformed = kd.optim.UseEmaParams().transform(state)

  chex.assert_trees_all_close(transformed.params, ema_weights)


def test_use_ema_params_with_explicit_transform_path_selects_target_ema_state():
  fast_ema_weights = {'w': jnp.array([5.0, 6.0])}
  slow_ema_weights = {'w': jnp.array([10.0, 20.0])}
  state = _make_train_state(
      params={'w': jnp.array([1.0, 2.0])},
      opt_state={
          'ema_fast': kd.optim.ema_params(decay=0.9).init(fast_ema_weights),
          'ema_slow': kd.optim.ema_params(decay=0.999).init(slow_ema_weights),
      },
  )

  transformed = kd.optim.UseEmaParams(
      ema_params_transform='ema_slow'
  ).transform(state)

  chex.assert_trees_all_close(transformed.params, slow_ema_weights)


def test_use_ema_params_without_ema_state_raises_value_error():
  state = _make_train_state(opt_state={'adam': {'m': jnp.array([0.0])}})

  with pytest.raises(ValueError, match='No EmaParamsState found'):
    kd.optim.UseEmaParams().transform(state)


def test_use_ema_params_with_non_ema_transform_path_raises_value_error():
  state = _make_train_state(opt_state={'adam': {'m': jnp.array([0.0])}})

  with pytest.raises(ValueError, match='not an instance of `EmaParamsState`'):
    kd.optim.UseEmaParams(ema_params_transform='adam').transform(state)


def test_use_ema_params_with_multiple_ema_states_raises_value_error():
  ema_init = kd.optim.ema_params(decay=0.9).init({'w': jnp.array([1.0])})
  state = _make_train_state(
      opt_state={'ema_fast': ema_init, 'ema_slow': ema_init},
  )

  with pytest.raises(ValueError, match='Found multiple EmaParamsStates'):
    kd.optim.UseEmaParams().transform(state)


def test_use_ema_params_with_masked_params_and_partial_not_ok_raises_key_error():
  optimizer = kd.optim.partial_updates(
      kd.optim.named_chain(
          sgd=optax.sgd(learning_rate=0.5),
          ema=kd.optim.ema_params(decay=0.5, debias=False),
      ),
      mask=kd.optim.select('trainable'),
  )
  params = {
      'trainable': {'w': jnp.array([2.0], dtype=jnp.float32)},
      'frozen': {'w': jnp.array([9.0], dtype=jnp.float32)},
  }
  state = _make_train_state(params=params, opt_state=optimizer.init(params))

  with pytest.raises(KeyError, match='No EMA params found for path'):
    kd.optim.UseEmaParams(partial_ok=False).transform(state)


def test_use_ema_params_with_partial_ok_preserves_frozen_params():
  optimizer = kd.optim.partial_updates(
      kd.optim.named_chain(
          sgd=optax.sgd(learning_rate=0.5),
          ema=kd.optim.ema_params(decay=0.5, debias=False),
      ),
      mask=kd.optim.select('trainable'),
  )
  params0 = {
      'trainable': {'w': jnp.array([2.0], dtype=jnp.float32)},
      'frozen': {'w': jnp.array([9.0], dtype=jnp.float32)},
  }
  grads = {
      'trainable': {'w': jnp.array([-2.0], dtype=jnp.float32)},
      'frozen': {'w': jnp.array([100.0], dtype=jnp.float32)},
  }
  opt_state0 = optimizer.init(params0)
  updates, opt_state1 = optimizer.update(grads, opt_state0, params0)
  state = _make_train_state(
      params=optax.apply_updates(params0, updates),
      opt_state=opt_state1,
  )

  transformed = kd.optim.UseEmaParams(partial_ok=True).transform(state)

  # trainable.w: params0=2.0, step1 update=+1.0 -> params1=3.0, ema=2.5.
  # frozen.w: remains 9.0.
  chex.assert_trees_all_close(
      transformed.params,
      {
          'trainable': {'w': jnp.array([2.5], dtype=jnp.float32)},
          'frozen': {'w': jnp.array([9.0], dtype=jnp.float32)},
      },
  )


def test_checkpointer_preemption_preserves_ema_state(tmp_path: pathlib.Path):
  batch1 = {
      'x': jnp.full((2, 3), 1.0, dtype=jnp.float32),
      'y': jnp.full((2, 2), 0.0, dtype=jnp.float32),
  }
  batch2 = {
      'x': jnp.full((2, 3), 2.0, dtype=jnp.float32),
      'y': jnp.full((2, 2), 1.0, dtype=jnp.float32),
  }
  optimizer = kd.optim.named_chain(
      adam=optax.scale_by_adam(),
      lr=optax.scale_by_learning_rate(0.1),
      ema=kd.optim.ema_params(decay=0.9, debias=True),
  )

  # Given an uninterrupted 2-step reference trajectory.
  ref_step = _make_trainstep(optimizer=optimizer, seed=42)
  ref_state = _init_train_state(ref_step)
  ref_state, _ = ref_step.step(ref_state, batch1)
  ref_state, _ = ref_step.step(ref_state, batch2)

  # When a trajectory is preempted after step 1, saved, and restored for step 2.
  ckpt_dir = tmp_path / 'preempt_workdir'
  run1_step = _make_trainstep(optimizer=optimizer, seed=42)
  state_step1 = _init_train_state(run1_step)
  state_step1, _ = run1_step.step(state_step1, batch1)
  with kd.ckpts.Checkpointer(workdir=ckpt_dir, save_interval_steps=1) as ckpt:
    ckpt.save(state_step1, step=int(state_step1.step))

  run2_step = _make_trainstep(optimizer=optimizer, seed=42)
  with kd.ckpts.Checkpointer(workdir=ckpt_dir, save_interval_steps=1) as ckpt:
    restored_state = ckpt.restore(
        _init_train_state(run2_step, skip_transforms=True),
        donate=False,
    )
  resumed_state, _ = run2_step.step(restored_state, batch2)

  # Then the resumed trajectory matches the uninterrupted trajectory exactly.
  assert int(resumed_state.step) == int(ref_state.step) == 2
  resumed_opt_state = cast(Any, resumed_state.opt_state)
  ref_opt_state = cast(Any, ref_state.opt_state)
  np.testing.assert_array_equal(
      resumed_opt_state['ema'].count,
      ref_opt_state['ema'].count,
  )
  chex.assert_trees_all_close(
      resumed_state.params, ref_state.params, rtol=0, atol=0
  )
  chex.assert_trees_all_close(
      resumed_opt_state['ema'].ema_params,
      ref_opt_state['ema'].ema_params,
      rtol=0,
      atol=0,
  )


def test_partial_kauldron_loader_maps_ema_params_into_model_params(
    tmp_path: pathlib.Path,
):
  src_workdir = tmp_path / 'src_workdir'
  optimizer = kd.optim.named_chain(
      sgd=optax.sgd(learning_rate=0.5),
      ema=kd.optim.ema_params(decay=0.9, debias=True),
  )
  src_state = _save_two_step_ema_checkpoint(src_workdir, optimizer=optimizer)
  src_opt_state = cast(Any, src_state.opt_state)

  with kd.ckpts.PartialKauldronLoader(
      workdir=src_workdir,
      new_to_old={'params': 'opt_state.ema.ema_params'},
  ) as eval_loader:
    eval_step = _make_trainstep(
        optimizer=optimizer,
        init_transform=eval_loader,
        seed=99,
    )
    eval_state = _init_train_state(eval_step, skip_optimizer=True)

  chex.assert_trees_all_close(
      eval_state.params,
      src_opt_state['ema'].ema_params,
      rtol=0,
      atol=0,
  )


def test_partial_kauldron_loader_restores_ema_opt_state_count_and_weights(
    tmp_path: pathlib.Path,
):
  src_workdir = tmp_path / 'src_workdir'
  optimizer = kd.optim.named_chain(
      sgd=optax.sgd(learning_rate=0.5),
      ema=kd.optim.ema_params(decay=0.9, debias=True),
  )
  src_state = _save_two_step_ema_checkpoint(src_workdir, optimizer=optimizer)
  src_opt_state = cast(Any, src_state.opt_state)

  with kd.ckpts.PartialKauldronLoader(
      workdir=src_workdir,
      new_to_old={'opt_state.ema': 'opt_state.ema'},
  ) as finetune_loader:
    finetune_step = _make_trainstep(
        optimizer=optimizer,
        init_transform=finetune_loader,
        seed=99,
    )
    finetune_state = _init_train_state(finetune_step)

  finetune_opt_state = cast(Any, finetune_state.opt_state)
  np.testing.assert_array_equal(
      finetune_opt_state['ema'].count,
      jnp.array(2, dtype=jnp.int32),
  )
  chex.assert_trees_all_close(
      finetune_opt_state['ema'].ema_params,
      src_opt_state['ema'].ema_params,
      rtol=0,
      atol=0,
  )
