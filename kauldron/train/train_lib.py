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

"""Training loop library."""

from __future__ import annotations

import contextlib
from typing import Optional, Tuple

from absl import logging
from clu import periodic_actions
from etils import epath
from etils import exm
import jax
import jax.numpy as jnp
from kauldron.evals import eval_impl
from kauldron.train import config_lib
from kauldron.train import flatboard_utils
from kauldron.train import metric_writer
from kauldron.train import timer as timer_module
from kauldron.train import train_step
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member
from kauldron.utils import utils
from kauldron.utils.sharding_utils import sharding  # pylint: disable=g-importing-member
import tensorflow as tf

# Jax config options
# Required for the `jax.Array` parallelization
jax.config.update("jax_threefry_partitionable", True)


def train_impl(
    trainer: config_lib.Trainer,
) -> Tuple[train_step.TrainState, Optional[train_step.Auxiliaries]]:
  """Implements of `Trainer.train`."""
  tf.config.experimental.set_visible_devices([], "GPU")

  status.log("Configuring ...")
  _ensure_workdir(trainer.workdir)
  flatboard_utils.add_flatboards(trainer)

  hooks = []
  if status.is_lead_host:
    hooks.append(trainer.profiler)
    if status.on_xmanager:
      hooks.append(
          periodic_actions.ReportProgress(
              num_train_steps=trainer.num_train_steps
          )
      )
  writer = metric_writer.KDMetricWriter(
      workdir=trainer.workdir, collection="train"
  )

  status.log("Initializing ...")
  trainstep = trainer.trainstep
  ckptr = trainer.checkpointer
  latest_step = ckptr.latest_step
  initial_step = 0 if latest_step is None else latest_step

  state = trainstep.init(
      elem_spec=trainer.train_ds.element_spec,
      restoring=latest_step is not None,
  )

  # Initialize CheckpointManager and attempt to restore.
  state = ckptr.restore(state, noop_if_missing=True)

  if initial_step == 0:
    writer.write_config(trainer.raw_cfg)
    writer.write_param_overview(initial_step, state.params)
    writer.write_element_spec(initial_step, trainer.train_ds.element_spec)
    writer.write_context_structure(initial_step, trainer)

  timer = timer_module.PerformanceTimer(
      initial_step_num=initial_step,
      initial_training_time_hours=float(state.training_time_hours),
      per_device_batch_size=trainer.train_ds.batch_size / jax.device_count(),
      global_batch_size=trainer.train_ds.batch_size,
  )

  status.log(f"Starting training loop at step {initial_step}")
  # NOTE: DO *NOT* CHANGE THE ORDER OF OPERATIONS IN THE TRAINING LOOP!
  total_steps = trainer.num_train_steps + 1
  if trainer.stop_after_steps is not None:
    total_steps = min(total_steps, initial_step + trainer.stop_after_steps)
  aux = None

  if not jax.config.jax_disable_jit:
    # Prevent implicit device transfer inside the train loop
    # Doc at: https://jax.readthedocs.io/en/latest/transfer_guard.html
    # This can be locally changed with `with jax.transfer_guard('allow'):`
    # TODO(epot): Activate this after https://github.com/google/jax/issues/16002
    guard = jax.transfer_guard("disallow")
  else:
    guard = contextlib.nullcontext()

  with guard:
    for i, batch in utils.enum_iter(
        trainer.train_ds.device_put(),
        init_step=initial_step,
        total_steps=total_steps,
        desc="train",
    ):
      with timer.exclude_from_step_stats():
        if ckptr.should_save(i):
          # Take the time after executing the last training step so that the
          # times logged and stored with the ckecpoint match.
          state = state.replace(
              training_time_hours=sharding.device_put(
                  timer.total_training_time_hours, sharding.REPLICATED
              )
          )
          ckptr.save_state(state, i)

        for evaluator in trainer.evals.values():
          evaluator.maybe_eval(
              step=i,
              state=state,
          )

      log_summaries = i % trainer.log_summaries_every == 0
      log_metrics = i % trainer.log_metrics_every == 0
      if not log_summaries and not log_metrics:
        state, aux = trainstep.step(state, batch)  # pylint: disable=unused-variable
        timer.finish_step()
      else:
        state, aux = trainstep.step(
            state,
            batch,
            return_losses=True,
            return_metrics=log_metrics,
            return_summaries=log_summaries,
        )

        timer.finish_step()
        performance_stats = {
            f"perf_stats/{k}": v for k, v in timer.log_stats(step_num=i).items()
        }

        # NOTE: ensure that evaluation metrics are computed from the OLD model
        # state *before* backprop gradients are applied.
        if status.is_lead_host:
          writer.write_step_metrics(
              step=i,
              aux=aux,
              schedules=trainer.schedules,
              model_with_aux=trainstep.model_with_aux,
              performance_stats=performance_stats,
              log_summaries=log_summaries,
          )

      for h in hooks:
        h(i)

  # Notify the eval job training is complete
  if exm.is_running_under_xmanager():
    exm.current_work_unit().add_tag(eval_impl.TRAIN_COMPLETE_TAG)

  _sync()
  # Returning the final state is convenient for interactive training in colab
  return state, aux


def _ensure_workdir(workdir: epath.PathLike):
  """Ensure workdir is set and exists."""
  workdir = epath.Path(workdir) if workdir else epath.Path()
  if workdir == epath.Path():
    raise ValueError("--workdir must be set when running on XManager.")

  logging.info("Creating workdir: %s", workdir)
  workdir.mkdir(parents=True, exist_ok=True)


def _sync():
  """Syncs hosts and empties async computation queue."""

  def _psync(x):
    return jax.lax.psum(x, "i")

  x = jnp.ones([jax.local_device_count()])
  return jax.pmap(_psync, "i")(x).block_until_ready()
