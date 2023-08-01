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

from collections.abc import Mapping
import dataclasses
from typing import Optional

from etils import edc
from etils import epath
import flax
from flax import linen as nn
from kauldron import data
from kauldron import losses
from kauldron import metrics
from kauldron import summaries
from kauldron.train import checkpointer as checkpointer_lib
from kauldron.train import evaluators
from kauldron.train import flatboard
from kauldron.train import rngs_lib
from kauldron.utils import config_util
from kauldron.utils import xmanager
import optax


class Config(config_util.BaseConfig):
  """Base config class.

  Attributes:
    seed: Seed for all rngs
    workdir: Root dir of the experiment (usually set by XManager)
    train_ds: Dataset used in training
    model: Flax linen module
    rng_streams: Flax rng streams to use **in addition** of the default
      (`params`, `dropout`, `default`). If any of `params`, `dropout`, `default`
      is set here, it will overwrite the default value.
    num_train_steps: Number of training steps. If `None`, train on the full
      dataset for the number of epoch specified in `train_ds`
    log_metrics_every: x
    log_summaries_every: x
    train_losses: x
    train_metrics: x
    train_summaries: x
    schedules: x
    optimizer: x
    checkpointer: x
    eval:
    flatboards:
    run: XManager runtime parameters (e.g. which target is the config using)
  """

  seed: int = 0
  # usually set by the launcher
  workdir: edc.AutoCast[epath.Path] = epath.Path()

  # TODO(epot): Replace by non-TF generic protocol
  train_ds: data.TFDataPipeline
  model: nn.Module
  num_train_steps: Optional[int] = None
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

  # XManager parameters
  run: xmanager.RunConfig = dataclasses.field(
      default_factory=xmanager.RunConfig
  )

  def __post_init__(self):
    # Eventually propagate the seed from the root config
    # Set rngs before eval, as the rngs is used in eval.
    if self.rng_streams.seed is None:
      object.__setattr__(
          self,
          'rng_streams',
          dataclasses.replace(self.rng_streams, seed=self.seed),
      )
    object.__setattr__(
        self, 'eval', dataclasses.replace(self.eval, base_cfg=self)
    )
