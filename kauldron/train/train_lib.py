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

import functools
from typing import Any, Optional, Sequence, Tuple

from absl import logging
from clu import periodic_actions
from etils import epath
import flax
import jax
import jax.numpy as jnp
from kauldron import konfig
from kauldron import metrics
from kauldron import summaries
from kauldron.train import config_lib
from kauldron.train import flatboard
from kauldron.train import metric_writer
from kauldron.train import timer as timer_module
from kauldron.train import train_step
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member
from kauldron.utils import core
from kauldron.utils import paths as paths_lib
from kauldron.utils import utils
from kauldron.utils.sharding_utils import sharding  # pylint: disable=g-importing-member
import ml_collections
import tensorflow as tf

# Jax config options
# Required for the `jax.Array` parallelization
jax.config.update("jax_threefry_partitionable", True)


def train(
    raw_cfg: config_lib.Config,
) -> Tuple[train_step.TrainState, train_step.Auxiliaries]:
  """DEPRECATED. Use `cfg.train()` instead."""
  print(
      "***DEPRECATED***: Calling `kd.train.train(cfg)` is deprecated. You can"
      " call `cfg.train()` directly."
  )
  cfg = konfig.resolve(raw_cfg)
  return cfg.train()


def train_impl(
    cfg: config_lib.Config,
) -> Tuple[train_step.TrainState, Optional[train_step.Auxiliaries]]:
  """Implements of `Config.train`."""
  tf.config.experimental.set_visible_devices([], "GPU")

  status.log("Configuring ...")
  ensure_workdir(cfg.workdir)
  add_flatboards(cfg)

  hooks = []
  if status.is_lead_host:
    hooks.append(cfg.profiler)
    if status.on_xmanager:
      hooks.append(
          periodic_actions.ReportProgress(num_train_steps=cfg.num_train_steps)
      )
  writer = metric_writer.KDMetricWriter(workdir=cfg.workdir, collection="train")

  status.log("Initializing ...")
  trainstep = cfg.trainstep

  state = trainstep.init(cfg.train_ds.element_spec)

  # Initialize CheckpointManager and attempt to restore.
  ckptr = cfg.checkpointer
  state = ckptr.restore(state, noop_if_missing=True)
  latest_step = ckptr.latest_step
  initial_step = 0 if not latest_step else latest_step

  if initial_step == 0:
    writer.write_config(cfg.raw_cfg)
  writer.write_param_overview(initial_step, state.params)
  writer.write_element_spec(initial_step, cfg.train_ds.element_spec)
  writer.write_context_structure(initial_step, cfg)

  timer = timer_module.PerformanceTimer(
      initial_step_num=initial_step,
      initial_training_time_hours=float(state.training_time_hours),
      per_device_batch_size=cfg.train_ds.batch_size / jax.device_count(),
      global_batch_size=cfg.train_ds.batch_size,
  )

  status.log(f"Starting training loop at step {initial_step}")
  # NOTE: DO *NOT* CHANGE THE ORDER OF OPERATIONS IN THE TRAINING LOOP!
  total_steps = cfg.num_train_steps + 1
  if cfg.stop_after_steps is not None:
    total_steps = min(total_steps, initial_step + cfg.stop_after_steps)
  aux = None

  # Prevent implicit device transfer inside the train loop
  # Doc at: https://jax.readthedocs.io/en/latest/transfer_guard.html
  # This can be locally changed with `with jax.transfer_guard('allow'):`
  # TODO(epot): Activate this after https://github.com/google/jax/issues/16002
  with jax.transfer_guard("disallow"):
    for i, batch in utils.enum_iter(
        cfg.train_ds,
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

        cfg.eval.maybe_eval(
            step=i,
            state=state,
        )

      log_summaries = i % cfg.log_summaries_every == 0
      log_metrics = i % cfg.log_metrics_every == 0
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
          write_summaries(
              writer=writer,
              step=i,
              aux=aux,
              schedules=cfg.schedules,
              model_with_aux=trainstep.model_with_aux,
              performance_stats=performance_stats,
              log_summaries=log_summaries,
          )

      for h in hooks:
        h(i)

  sync()
  # Returning the final state is convenient for interactive training in colab
  return state, aux


def write_summaries(
    *,
    writer: metric_writer.KDMetricWriter,
    step,
    aux: train_step.Auxiliaries,
    schedules,
    model_with_aux: train_step.ModelWithAux,
    log_summaries,
    performance_stats: Optional[dict[str, float]] = None,
):
  """Logs scalar and image summaries."""
  # train losses
  loss_values = compute_and_flatten_summaries(
      model_with_aux.losses, states=aux.loss_states, prefix="losses"
  )
  # Multi-process communication
  with jax.spmd_mode("allow_all"), jax.transfer_guard("allow"):
    total_loss = jnp.sum(jnp.asarray(list(loss_values.values())))
  loss_values["losses/total"] = total_loss

  # train metrics
  metric_values = compute_and_flatten_summaries(
      model_with_aux.metrics, states=aux.metric_states, prefix="metrics"
  )
  # schedules
  schedule_values = compute_and_flatten_summaries(
      schedules,
      states=None,
      prefix="schedules",
      compute_fn=functools.partial(_compute_schedule, step=step),
  )
  performance_stats = performance_stats or {}
  with jax.transfer_guard("allow"):
    writer.write_scalars(
        step=step,
        scalars=(
            loss_values | metric_values | schedule_values | performance_stats
        ),
    )

  if log_summaries:
    with jax.transfer_guard("allow"):
      # image summaries  # TODO(klausg): unify with metrics
      image_summaries = {
          name: summary.get_images(**aux.summary_kwargs[name])
          for name, summary in model_with_aux.summaries.items()
          if isinstance(summary, summaries.ImageSummary)
      }
    # Throw an error if empty arrays are given. TB throws very odd errors
    # and kills Colab runtimes if we don't catch these ourselves.
    for name, image in image_summaries.items():
      if image.size == 0:
        raise ValueError(
            f"Image summary `{name}` is empty array of shape {image.shape}."
        )
    writer.write_images(step=step, images=image_summaries)

    # histograms
    hist_summaries = {
        name: summary.get_tensor(**aux.summary_kwargs[name])
        for name, summary in model_with_aux.summaries.items()
        if isinstance(summary, summaries.HistogramSummary)
    }
    for name, (_, tensor) in hist_summaries.items():
      if tensor.size == 0:
        raise ValueError(
            f"Histogram summary `{name}` is empty array of shape"
            f" {tensor.shape}."
        )
    writer.write_histograms(
        step=step,
        arrays={k: tensor for k, (_, tensor) in hist_summaries.items()},
        num_buckets={
            k: n_buckets for k, (n_buckets, _) in hist_summaries.items()
        },
    )

  writer.flush()


def _compute_metric(metric: metrics.Metric, state: Any):
  """Compute the value of a metric for a given state and return the result."""
  # Accept cross-process computation (some metrics cannot be jitted)
  with jax.spmd_mode("allow_all"), jax.transfer_guard("allow"):
    return metric.compute(state)


def _compute_schedule(sched, step: int):
  """Evaluate schedule for step and return result."""
  with jax.transfer_guard("allow"):
    return sched(step)


def compute_and_flatten_summaries(
    summarizers: Any,
    states: Optional[Any] = None,
    prefix: Optional[str] = None,
    compute_fn=_compute_metric,
) -> dict[str, float]:
  """Compute a flat dictionary of metric values for reporting to tensorboard.

  Takes a nested structure of summarizers and corresponding states, evaluates
  them and flattens it into a dict[str, float] with paths of the form
  "metrics/recon/psnr".

  Args:
    summarizers: A (nested) dictionary of summary computing objects (e.g.
      losses, metrics, schedules). The leaf-values of this structure will be
      passed as the first positional argument to compute_fn.
    states: An optional (nested) dictionary of states. If this is not None then
      it has to have the same structure as summarizers and its values will be
      passed as a second pos-arg to compute_fn.
    prefix: An optional prefix path to add to the output keys.
    compute_fn: A callable for computing the values for all the leafs. If states
      is None then it should take the form (summarizer,) -> float otherwise
      (summarizer, state) -> float.

  Returns:
    A flat dictionary mapping paths of the form "losses/mse" to float values.
    This can e.g. be used for writing summaries to tensorboard.
  """
  if isinstance(summarizers, ml_collections.ConfigDict):
    # Convert to dict because tree_map doesn't handle ConfigDicts properly
    summarizers = summarizers.to_dict()
  if not isinstance(summarizers, flax.core.FrozenDict):
    summarizers = flax.core.FrozenDict(summarizers)
  if states is not None and not isinstance(states, flax.core.FrozenDict):
    states = flax.core.FrozenDict(states)
  if states is None:
    values = jax.tree_util.tree_map(compute_fn, summarizers)
  else:
    values = jax.tree_util.tree_map(compute_fn, summarizers, states)

  def _format_path(path, prefix: Optional[str] = None):
    str_parts = [prefix] if prefix else []
    str_parts.extend([paths_lib.jax_key_entry_to_str(p) for p in path])  # pylint: disable=no-value-for-parameter
    return "/".join(str_parts)

  flat_values, _ = jax.tree_util.tree_flatten_with_path(values)
  flat_results = {
      _format_path(path, prefix=prefix): value for path, value in flat_values
  }
  return flat_results


def tree_flatten_with_slash_path(config_dict) -> dict[str, Any]:
  """Turn the (nested) keys of a config dict into a list of 'keys/like/this'."""
  if isinstance(config_dict, ml_collections.ConfigDict):
    # Normalize ConfigDict to dict
    config_dict = config_dict.to_dict()
  flat_tree_items, _ = jax.tree_util.tree_flatten_with_path(config_dict)
  return {
      "/".join(core.Path.from_jax_path(jax_path).parts): value
      for jax_path, value in flat_tree_items
  }


def get_loss_y_keys(config) -> Sequence[str]:
  """Get a list of loss-keys for a given config."""
  # train losses
  loss_names = {k for k in tree_flatten_with_slash_path(config.train_losses)}
  # evaluator losses
  for evaluator in config.eval.flatten():
    # TODO(epot): Cleaner way to support metrics for custom evaluator
    if not hasattr(evaluator, "losses"):
      continue
    loss_names |= {k for k in tree_flatten_with_slash_path(evaluator.losses)}

  # If more than one loss, add the total loss
  if len(loss_names) > 1:
    loss_names = {"total"} | loss_names
  return [f"losses/{l.replace('.', '/')}" for l in loss_names]


def get_metric_y_keys(config) -> Sequence[str]:
  """Get a list of metric-keys for a given config."""
  metric_names = {k for k in tree_flatten_with_slash_path(config.train_metrics)}
  # add evaluator metrics
  for evaluator in config.eval.flatten():
    if not hasattr(evaluator, "metrics"):
      continue
    metric_names |= {k for k in tree_flatten_with_slash_path(evaluator.metrics)}
  return [f"metrics/{l.replace('.', '/')}" for l in sorted(metric_names)]


def get_schedule_y_keys(config) -> Sequence[str]:
  """Get a list of schedule-keys for a given config."""
  schedule_names = [k for k in tree_flatten_with_slash_path(config.schedules)]
  return [f"schedules/{l.replace('.', '/')}" for l in schedule_names]


def get_perf_stat_y_keys() -> Sequence[str]:
  """Get a list of performance statistics keys."""
  return [
      f"perf_stats/{y}"  # pylint: disable=g-complex-comprehension
      for y in [
          "steps_per_sec",
          "data_points_per_sec_global",
          "data_points_per_sec_per_device",
          "total_training_time_hours",
      ]
  ]


def get_data_collections(config: Any) -> list[str]:
  """Return a list of datatable collections for a given resolved config."""
  collections = ["train"]
  evaluator = config.eval
  if evaluator and hasattr(evaluator, "name"):
    collections.append(evaluator.name)
  elif evaluator and hasattr(evaluator, "children"):
    collections.extend(eval.name for eval in evaluator.children)

  return collections


def get_default_dashboards(config):
  """Return the default set of Flatboard dashboards for given config."""
  data_collections = get_data_collections(config)
  dashboards = {}
  # losses
  y_keys = get_loss_y_keys(config)
  if y_keys:
    dashboards["losses"] = flatboard.DefaultDashboard(
        title="{xid}: Losses", y_keys=y_keys, collections=data_collections
    )

  # metric
  y_keys = get_metric_y_keys(config)
  if y_keys:
    dashboards["metrics"] = flatboard.DefaultDashboard(
        title="{xid}: Metrics", y_keys=y_keys, collections=data_collections
    )

  # schedules
  y_keys = get_schedule_y_keys(config)
  if y_keys:
    dashboards["schedules"] = flatboard.DefaultDashboard(
        title="{xid}: Schedules",
        y_keys=y_keys,
        collections=["train"],
    )

  # perf_stats
  y_keys = get_perf_stat_y_keys()
  if y_keys:
    dashboards["perf_stats"] = flatboard.DefaultDashboard(
        title="{xid}: Performance Statistics",
        y_keys=y_keys,
        collections=["train"],
    )
  return dashboards


def ensure_workdir(workdir: epath.PathLike):
  """Ensure workdir is set and exists."""
  workdir = epath.Path(workdir) if workdir else epath.Path()
  if workdir == epath.Path():
    raise ValueError("--workdir must be set when running on XManager.")

  logging.info("Creating workdir: %s", workdir)
  workdir.mkdir(parents=True, exist_ok=True)


def add_flatboards(cfg):
  """Add flatboards based on cfg.flatboards or default flatboards."""
  if not status.on_xmanager or not status.is_lead_host or status.wid != 1:
    return  # only add flatboards once per experiment
  dashboard_factories = cfg.flatboards
  if not dashboard_factories:
    dashboard_factories = get_default_dashboards(cfg)
  flatboard.add_flatboard_artifacts(dashboard_factories)


def sync():
  """Syncs hosts and empties async computation queue."""

  def _sync(x):
    return jax.lax.psum(x, "i")

  x = jnp.ones([jax.local_device_count()])
  return jax.pmap(_sync, "i")(x).block_until_ready()
