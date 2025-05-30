# Copyright 2025 The kauldron Authors.
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

"""Evaluator that runs a trainer. Useful for training readout heads."""

from __future__ import annotations

import copy
import dataclasses
import functools
import os
import typing
from typing import Any, Callable, Optional

from absl import logging
import chex
from clu import metric_writers
from etils import epath
import flax.core
import jax
from kauldron import kd
from kauldron import kontext
from kauldron.checkpoints import checkpointer as checkpointer_lib
from kauldron.contrib.evals import train_eval
from kauldron.evals import run_strategies
from kauldron.train import metric_writer
from kauldron.train import setup_utils
from kauldron.train import trainer_lib
from kauldron.utils import kdash
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member
import orbax.checkpoint as ocp


def default_stop_after_steps_schedule_fn(
    step: int, default_num_train_evaluator_steps: int
) -> int:
  """Default schedule for the number of train_evaluator steps."""
  del step
  return default_num_train_evaluator_steps


class TrainEvaluator(kd.evals.EvaluatorBase):
  """Evaluator that runs a trainer from a standalone config.

  Usage:
  ```
  cfg.evals = {
      "readout": kd.contrib.evals.TrainEvaluator(
          run=kd.evals.EveryNSteps(100),
          readout_config=kd.contrib.examples.readout_mnist.get_config(),
      ),
  ```

  Attributes:
    readout_config: Should be set to the config of a standalone trainer like
      `kauldron/contrib/examples/mnist_readout.py` which uses a ReadoutWrapper
      module.
    model: The model to evaluate. Automatically set to the model of the parent
      trainer.
    submodule_to_swap_path: The path of the submodule within the readout_config
      which should be swapped out for the model of the main config. Defaults to
      "model.model" which is the correct path if the readout_config uses (just)
      a ReadoutWrapper module to wrap the main model.
    subparams_to_swap_path: The path of to the parameters of the submodule
      within the readout_config which should be swapped out for the parameters
      of model of the main config. Defaults to the value of
      submodule_to_swap_path.
    stop_after_steps_schedule_fn: A function that takes the current pre-training
      step of the evaluated checkpoint and the default number of train_evaluator
      steps and returns the number of train_evaluator steps after which the
      evaluator should stop. This is subtly different from setting the number of
      train_evaluator steps, because the `stop_after_steps` parameter does not
      influence any other settings in the evaluator that depend on the number of
      steps e.g. the learning rate schedule.
    overrides: A dictionary of overrides to be applied to the model. This can be
      used to e.g. try out different masking ratio in MAE. Does not support
      multiple overrides of a same hyperparameter like in sweeping. Does not
      support changing parameters of the model. The "model." prefix is not
      needed (e.g. processor.drop_ratio instead of model.processor.drop_ratio).

  Usually submodule_to_swap_path and subparams_to_swap_path are set to the same
  value. However, in some cases they can be different. For instance:
  ```
  class Wrapper(nn.Module):
    pretrained_model: nn.Module
    def setup():
      self.model = self.pretrained_model.copy()
  ```
  In this situation submodule_to_swap_path must be set to
  "model.pretrained_model" and whereas subparams_to_swap_path must be set to
  "model.model", because the parameters of the wrapper are {"model": ...}
  whereas the module we need to replace in the config is called
  pretrained_model.

  Limitations:
    - Does not report any of the training metrics / losses / summaries. Only the
      metrics reported by running the evaluators of the readout_config once at
      the end after training.
    - Does not store any of the weights / checkpoints.
    - Might fail if the model of the parent trainer uses any config references.
    - The metrics of this evaluator are not included in the default flatboard.
  """

  readout_config: kd.konfig.ConfigDict

  submodule_to_swap_path: str = "model.model"
  subparams_to_swap_path: str | None = None
  stop_after_steps_schedule_fn: Callable[[int, int], int] = (
      default_stop_after_steps_schedule_fn
  )
  overrides: dict[str, Any] | None = None

  readout_inputs: Optional[Any] = None

  __konfig_resolve_exclude_fields__ = ("readout_config",)

  def evaluate(
      self, state: kd.train.TrainState, step: int
  ) -> kd.train.AuxiliariesState:
    """Run one full evaluation."""
    self._assert_root_cfg_resolved()

    ## ================= Run trainer =================
    trainer = self.readout_trainer_for_step(state, step)

    # Update the flatboard collection names to the new step.
    self._update_flatboard(step)

    # Train the readout head.
    with jax.transfer_guard("allow"):
      train_state, _ = trainer.train()

    # ================ Evaluate on the trained state =====================

    for sub_evaluator in trainer.evals.values():
      # Override the metric writer with the one of the parent trainer.
      sub_evaluator = dataclasses.replace(
          sub_evaluator,
          writer=self.writer,
          name=self._global_collection_name(sub_evaluator.name),
      )
      # Do not compute losses when possible to avoid cluttering the main
      # flatboard
      if hasattr(sub_evaluator, "losses"):
        sub_evaluator = sub_evaluator.replace(losses=flax.core.freeze({}))

      sub_evaluator.evaluate(state=train_state, step=step)

    # We do not return auxiliaries. We could try to merge all auxiliaries from
    # all sub-evaluators, but it's not clear how to handle conflicts (if two
    # evaluators use same metric name).
    return kd.train.AuxiliariesState()

  @functools.cached_property
  def readout_trainer_base(self) -> trainer_lib.Trainer:
    readout_cfg = copy.copy(self.readout_config)
    readout_cfg = typing.cast(trainer_lib.Trainer, readout_cfg)

    # Avoid problems with workdir=None
    # Should NOT be used!
    readout_cfg.workdir = os.fspath(
        self.base_cfg.workdir / f"train_evals.{self.name}"  # pytype: disable=unsupported-operands
    )

    # TODO(epot): Supports metrics. Everytime `.evaluate` is run, metrics
    # should use a new directory

    # Explicitly disable some options (deleting the field will
    # use the default values)
    readout_cfg.xm_job = None
    readout_cfg.profiler = None
    readout_cfg.writer = None
    # TODO(klausg) retain other init transforms?
    # TODO(epot): Dangerous to remove init_transform. Instead should raise
    # an explicit error.
    readout_cfg.init_transform = None

    build_ctx = self.base_cfg.setup.flatboard_build_context
    with kd.konfig.mock_modules():
      # Overwrite the setup to disable creating new workdir, setup flatboard,
      # etc.
      readout_cfg.setup = train_eval._EvalSetup(  # pylint: disable=protected-access
          tqdm_info=kd.train.TqdmInfo(
              desc=f"train_eval.{self.name}",
              log_xm=False,
          ),
          # Forward other arguments from the main `Trainer.setup`
          add_flatboard=self.base_cfg.setup.add_flatboard,
          flatboard_build_context=kdash.BuildContext(
              collection_path_prefix=build_ctx.collection_path_prefix,
              sweep_argnames=build_ctx.sweep_argnames,
          ),
      )

    # WARNING: using the model config might fails when it contains references
    submodule_to_swap_path = kd.kontext.Path.from_str(
        self.submodule_to_swap_path
    )
    # Make sure submodule_to_swap already exists in the config
    submodule_to_swap = submodule_to_swap_path.get_from(readout_cfg)
    assert submodule_to_swap is not None
    # Swap out the submodule
    assert self.base_cfg.raw_cfg is not None
    submodule_to_swap_path.set_in(readout_cfg, self.base_cfg.raw_cfg.model)

    if self.overrides is not None:
      for k, v in self.overrides.items():
        kd.kontext.Path.from_str(self.submodule_to_swap_path + "." + k).set_in(
            readout_cfg, v
        )

    if self.readout_inputs is not None:
      readout_cfg.model.readout_inputs = dict(self.readout_inputs)

    return kd.konfig.resolve(readout_cfg)

  def readout_trainer_for_step(
      self, state: kd.train.TrainState, step: int
  ) -> trainer_lib.Trainer:
    """Returns the readout trainer customized for the given step."""
    subparams_to_swap_path = self.subparams_to_swap_path
    if subparams_to_swap_path is None:
      subparams_to_swap_path = self.submodule_to_swap_path

    checkpointer = self.readout_trainer_base.checkpointer
    # Reuse trainer checkpointer, but add the name and step-suffix to the
    # workdir.
    if isinstance(checkpointer, checkpointer_lib.Checkpointer):
      workdir = checkpointer.workdir / f"_{self.name}_{step}"
      (workdir / checkpointer_lib.CHECKPOINT_FOLDER_NAME).mkdir(
          parents=True, exist_ok=True
      )
      logging.info("Create checkpointer with: workdir: %s", workdir)
      checkpointer = dataclasses.replace(
          checkpointer,
          workdir=workdir,
          max_to_keep=1,
          create=False,
          # The checkpointer uses a separate workdir for each step,
          # hence it is safe to use a separate barrier sync key as well.
          # Sometimes saving checkpoints can take a long time, so using separate
          # barrier sync keys for each step avoids false-positive errors from
          # the barrier sync, when a previous step is still being saved.
          multiprocessing_options=ocp.options.MultiprocessingOptions(
              barrier_sync_key_prefix=f"{self.name}_{step}"
          ),
      )
    else:
      raise ValueError(f"Unsupported checkpointer type: {type(checkpointer)}")
    return dataclasses.replace(
        self.readout_trainer_base,
        workdir=workdir,
        # Initialize the model with the params from the parent trainer.
        init_transform=_OverwriteParams(
            copy_params=self._copy_params_when_overwriting,
            subparams_to_swap_path=subparams_to_swap_path,
            state_from_training=state,
        ),
        checkpointer=checkpointer,
        # Write the metrics in a separate `{collection}.{step}`
        writer=_EvalWriter(  # pylint: disable=protected-access
            add_artifacts=False,
            curr_step=step,
            train_eval_name=self.name,
        ),
        stop_after_steps=self.stop_after_steps_schedule_fn(
            step, self.readout_trainer_base.num_train_steps
        ),
    )

  @property
  def _copy_params_when_overwriting(self) -> bool:
    """Whether to copy the `state` passed in `TrainEvaluator.evaluate`.

    The train state is passed to `trainstep.step` which donates the params,
    so we need to make a copy before.
    This can only be skipped when the eval is run as a standalone job
    (without other evals), as the state is recreated from the checkpoint
    at every iterations.

    Returns:
      Whether to copy the `state` passed to `TrainEvaluator.evaluate`.
    """
    if (
        isinstance(self.run, run_strategies.Standalone)
        and not self.run.job_group
    ):
      return False
    else:
      return True

  @functools.cached_property
  def __dashboards__(
      self,
  ) -> kdash.DashboardsBase:

    def _add_prefix(db: kdash.SingleDashboard):
      new_plots = [
          dataclasses.replace(
              p,
              collections=[
                  self._global_collection_name(c) for c in p.collections
              ],
          )
          for p in db.plots
      ]
      return dataclasses.replace(db, plots=new_plots)

    # Add the main collections
    all_dashboards = []
    for evaluator in self.readout_trainer_base.evals.values():
      dashboards = evaluator.__dashboards__.normalize().dashboards
      # Remove the losses and perf dashboard to not clutter the main flatboard.
      dashboards.pop("losses", None)
      dashboards.pop("perf_stats", None)
      # Update the collection name to add the `self.name` prefix to all
      # sub-evaluators.
      all_dashboards.extend(_add_prefix(db) for db in dashboards.values())

    # Add the extra dashboard containing the training metrics.
    all_dashboards.append(self._get_extra_dashboard([]))

    return kdash.MultiDashboards.from_iterable(all_dashboards)

  def _get_extra_dashboard(self, steps: list[int]) -> kdash.SingleDashboard:

    train_collections = [
        _collection_name_with_step(self.name, "train", step) for step in steps
    ]

    def _make_collections(collections_: list[str]) -> dict[str, list[str]]:
      eval_collections = {
          c: [_collection_name_with_step(self.name, c, step) for step in steps]
          for c in collections_
      }
      # ` train` rather than `train` to ensure it is displayed first.
      return {
          " train": train_collections,
          **eval_collections,
      }

    all_plots = []
    for evaluator in self.readout_trainer_base.evals.values():
      all_dashboards = evaluator.__dashboards__.normalize().dashboards
      # Do not report the perfs stats.
      all_dashboards.pop("perf_stats", None)

      # Merge all plots from the sub-evaluators to display them in a single
      # new dashboard.
      for db in all_dashboards.values():
        for plot in db.plots:
          # TODO(epot): Add facet between train/eval
          plot = dataclasses.replace(
              plot,
              # The collections are dynamically updated at every `.evaluate()`
              # step
              collections=[],
              facet_to_collections=_make_collections(plot.collections),
          )
          all_plots.append(plot)

    return kdash.SingleDashboard(
        name=self._extra_dashboard_name,
        title=f"{{xid}}: TrainEval {self.name}",
        plots=all_plots,
    )

  @functools.cached_property
  def _extra_dashboard_name(self) -> str:
    return f"train_eval.{self.name}"

  def _global_collection_name(self, sub_eval_name: str) -> str:
    """Metrics displayed along the parent trainer."""
    return f"{self.name}.{sub_eval_name}"

  def _update_flatboard(self, step: int) -> None:
    """Update the flatboard collections."""
    if not status.on_xmanager or not status.is_lead_host:
      return  # only add flatboards once per experiment

    # Keep track of all previous steps by writing a txt file.
    if self._steps_path.exists():
      all_steps = self._steps_path.read_text().split()
      all_steps = [int(x) for x in all_steps]
    else:
      all_steps = []
    all_steps.append(step)
    self._steps_path.write_text("\n".join(str(x) for x in all_steps))

    dashboard = self._get_extra_dashboard(all_steps)
    dashboard = dashboard.build(self.base_cfg.setup.flatboard_build_context)

  @functools.cached_property
  def _steps_path(self) -> epath.Path:
    """File containing the list of steps."""
    return self.base_cfg.workdir / f"train_eval-{self.name}-steps.txt"  # pytype: disable=unsupported-operands


# TrainEvaluator has custom `cfg` objects:
# ================ cfg.init_transform ================


@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class _OverwriteParams(kd.ckpts.AbstractPartialLoader):
  """Overwrite the params of the state to train in the readout...

  ...with the params from the global state from the parent training loop.
  """

  copy_params: bool
  subparams_to_swap_path: str

  # To avoid triggering new `jax.jit` compilation (in `TrainStep.step`), we
  # set `eq=False`/`hash=False`, so the jit hash do not depend on this field.
  state_from_training: kd.train.TrainState | None = dataclasses.field(
      compare=False, hash=False
  )

  def transform(self, state: kd.train.TrainState) -> kd.train.TrainState:
    params_key = ".".join(self.subparams_to_swap_path.split(".")[1:])
    params = dict(state.params)
    assert self.state_from_training is not None
    state_params = self.state_from_training.params
    if self.copy_params:
      state_params = jax.tree.map(lambda x: x.copy(), state_params)
    # Check that the params structure of the readout is the same as the params
    # structure of the parent trainer, if that's not the case, something
    # went wrong and we should raise an error.
    chex.assert_trees_all_equal_structs(
        kontext.get_by_path(params, params_key), state_params
    )
    kontext.set_by_path(params, params_key, state_params)
    return state.replace(params=params)


# ================ cfg.writer ================


# TODO(epot): Is it possible to merge use `KDMetricWriter` directly? By adding
# some `collection_prefix`, `collection_suffix`,...
@dataclasses.dataclass(kw_only=True, eq=True, frozen=True)
class _EvalWriter(
    metric_writer.NoopMetadataWriter,
    metric_writer.KDMetricWriter,
):
  """Custom writer which tracks the additional training steps."""

  curr_step: int = 0
  train_eval_name: str = ""

  @property
  def _collection_name_with_step(self) -> str:
    return _collection_name_with_step(
        self.train_eval_name, self.collection, self.curr_step
    )

  @functools.cached_property
  def _scalar_datatable_name(self) -> str:
    self._assert_collection_is_set()
    return f"{self._collection_path_prefix}{self._collection_name_with_step}"

  @functools.cached_property
  def _log_writer(self) -> metric_writers.MetricWriter:
    self._assert_collection_is_set()
    # Write the metrics in a separate `{collection}.{step}`
    return metric_writers.AsyncWriter(
        metric_writers.LoggingWriter(self._collection_name_with_step)
    )

  @functools.cached_property
  def _scalar_writer(self) -> metric_writers.MetricWriter:
    if status.on_xmanager and status.is_lead_host:
      return metric_writers.AsyncWriter(
          metric_writers.DatatableWriter(
              datatable_name=self._scalar_datatable_name,
              keys=[("wid", status.wid)],
          ),
      )
    else:
      return self._noop

  @functools.cached_property
  def _tf_summary_writer(self) -> metric_writers.MetricWriter:
    # No TF summaries for evals
    return self._noop

  @functools.cached_property
  def _array_writer(self) -> metric_writers.MetricWriter:
    # No array summaries for evals
    return self._noop


def _collection_name_with_step(
    train_eval_name: str, collection: str, step: int
) -> str:
  return f"{train_eval_name}.{collection}.{step}"


# ================ cfg.setup ================


class _EvalSetup(setup_utils.Setup):

  def run(self, trainer: trainer_lib.Trainer) -> None:
    # Skip all the setup (workdir, flatboard dashboard,...)
    pass

  def log_status(self, msg: str) -> None:
    logging.info(msg, stacklevel=2)


# ================ cfg.stop_after_steps_schedule_fn ================


def linear_stop_after_steps_schedule_fn(
    step: int,
    default_num_train_evaluator_steps: int,
    ramp_up_for_num_pre_training_steps: int = 100_000,
) -> int:
  """Linear schedule for the number of evaluator train steps."""
  return min(
      max(
          int(
              default_num_train_evaluator_steps
              * step
              / ramp_up_for_num_pre_training_steps
          ),
          1,
      ),
      default_num_train_evaluator_steps,
  )
