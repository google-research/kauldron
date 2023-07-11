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
from kauldron.utils import config_util
from kauldron.utils import xmanager
import optax


class Config(config_util.BaseConfig):
  """Base config class."""

  seed: int
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
  checkpointer: checkpointer_lib.Checkpointer

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
    object.__setattr__(
        self, 'eval', dataclasses.replace(self.eval, base_cfg=self)
    )
