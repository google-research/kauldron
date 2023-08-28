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

"""Config."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, Optional

from etils import edc
from etils import epath
import flax
from flax import linen as nn
from kauldron import data
from kauldron import konfig
from kauldron import losses
from kauldron import metrics
from kauldron import summaries
from kauldron.checkpoints import checkpointer as checkpointer_lib
from kauldron.train import evaluators
from kauldron.train import flatboard
from kauldron.train import rngs_lib
from kauldron.train import train_lib
from kauldron.train import train_step
from kauldron.utils import config_util
from kauldron.utils import xmanager
import optax


# TODO(epot): Rename: Experiment, Plan, Main, Root, Trainer, Train ?


class Config(config_util.BaseConfig):
  """Base config class.

  Attributes:
    seed: Seed for all rngs
    workdir: Root dir of the experiment (usually set by XManager)
    aux: Dictionary of arbitrary values to be referenced elsewhere
    train_ds: Dataset used in training
    eval_ds: Dataset used in eval (see https://kauldron.rtfd.io/en/latest/eval.html to activate eval)
    model: Flax linen module
    rng_streams: Flax rng streams to use **in addition** of the default
      (`params`, `dropout`, `default`). If any of `params`, `dropout`, `default`
      is set here, it will overwrite the default value.
    num_train_steps: Number of training steps. If `None`, train on the full
      dataset for the number of epoch specified in `train_ds`
    stop_after_steps: Optionally stop already after running this many steps. If
      set, overwrite `num_train_steps`. Allow to debug on Colab without
      modifying the learning rate schedules and other values that depend on
      `num_train_steps`.
    log_metrics_every: x
    log_summaries_every: x
    train_losses: x
    train_metrics: x
    train_summaries: x
    schedules: x
    optimizer: x
    checkpointer: x
    eval: Evaluator to use (e.g. `kd.train.SingleEvaluator`)
    flatboards: x
    trainstep: Training loop step. Do not set this field unless you need a
      custom training step.
    run: XManager runtime parameters (e.g. which target is the config using)
    raw_cfg: Original config from which this `Config` was created. Automatically
      set during `konfig.resolve()`
  """

  seed: int = 0
  # usually set by the launcher
  workdir: edc.AutoCast[epath.Path] = epath.Path()
  aux: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  # TODO(epot): Replace by non-TF generic protocol
  train_ds: data.TFDataPipeline
  eval_ds: data.TFDataPipeline = None
  model: nn.Module
  num_train_steps: Optional[int] = None
  stop_after_steps: Optional[int] = None
  log_metrics_every: int = 100
  log_summaries_every: int = 1000
  train_losses: Mapping[str, losses.Loss] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  train_metrics: Mapping[str, metrics.Metric] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  train_summaries: Mapping[str, summaries.Summary] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  schedules: Mapping[str, optax.Schedule] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  optimizer: optax.GradientTransformation
  checkpointer: checkpointer_lib.BaseCheckpointer = dataclasses.field(
      default_factory=checkpointer_lib.NoopCheckpointer
  )

  rng_streams: rngs_lib.RngStreams = dataclasses.field(
      default_factory=rngs_lib.RngStreams
  )

  eval: evaluators.EvaluatorBase = dataclasses.field(
      default_factory=evaluators.NoopEvaluator
  )
  flatboards: Mapping[str, flatboard.DashboardFactory] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )

  trainstep: train_step.TrainStep = dataclasses.field(
      default_factory=train_step.TrainStep, repr=False
  )

  # XManager parameters
  run: xmanager.RunConfig = dataclasses.field(
      default_factory=xmanager.RunConfig
  )

  raw_cfg: Optional[konfig.ConfigDict] = dataclasses.field(
      default=None, repr=False
  )

  def __post_init__(self):
    # Some config object values are lazy-initialized from the root config.
    # See `UpdateFromRootCfg` for details
    for attr_name in (
        'train_ds',
        'rng_streams',
        'eval',
        'checkpointer',
        'trainstep',
    ):
      if hasattr(self, attr_name):
        object.__setattr__(
            self, attr_name, getattr(self, attr_name).update_from_root_cfg(self)
        )

  def __post_konfig_resolve__(self, cfg: konfig.ConfigDict) -> None:
    """Bind the raw config to kd. Called during `kd.konfig.resolve()`."""
    # TODO(epot): Should freeze and deep-copy the config, but let's do this
    # after fiddle migration.
    object.__setattr__(self, 'raw_cfg', cfg)

  # Do not use property to make it explicit this is recomputed each time
  def init_state(self) -> train_step.TrainState:
    """Create the state: `cfg.trainstep.init(cfg.train_ds.element_spec)`."""
    return self.trainstep.init(self.train_ds.element_spec)

  def train(self) -> tuple[train_step.TrainState, train_step.Auxiliaries]:
    """Main method that train/evaluate the object.

    Similar to:

    ```python
    state = trainer.init_state().replicate()

    for batch in trainer.train_ds:
      batch = trainer.trainstep.step(batch, state)
    ```

    Returns:
      Final model state
      Auxiliaries
    """
    return train_lib.train_impl(self)
