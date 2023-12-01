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

"""Trainer."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import functools
import typing
from typing import Any, Optional

from etils import edc
from etils import epath
import flax
from flax import linen as nn
import jax
from kauldron import data
from kauldron import konfig
from kauldron import losses
from kauldron import metrics
from kauldron import summaries
from kauldron.checkpoints import checkpointer as checkpointer_lib
from kauldron.data import utils as data_utils
from kauldron.evals import evaluators
from kauldron.train import flatboard
from kauldron.train import rngs_lib
from kauldron.train import train_lib
from kauldron.train import train_step
from kauldron.utils import config_util
from kauldron.utils import context as context_lib
from kauldron.utils import profile_utils
import optax

# TODO(epot): Maybe merge like `konfig.imports(lazy=['*'])`
with (
    konfig.set_lazy_imported_modules(),
    konfig.imports(),
):
  # Do not resolve job_lib to not link the full XManager API to Kauldron
  from kauldron.xm._src import job_lib  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


# TODO(epot): There's some strange interaction between `get_type_hints` from
# `edc`, `ConfigDictLike` and lazy job imports. Should investigate but as it's
# not critical, just use this workaround.
if typing.TYPE_CHECKING:
  _JobConfigDict = konfig.ConfigDictLike[job_lib.Job]
else:
  _JobConfigDict = Any


class Trainer(config_util.BaseConfig):
  """Base config class.

  Attributes:
    seed: Seed for all rngs
    workdir: Root dir of the experiment (usually set by XManager)
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
    evals: Evaluators to use (e.g. `{'eval': kd.eval.Evaluator()}`)
    flatboards: x
    profiler: Profiler can be customized (see `kd.inspect.Profile`)
    aux: Arbitrary additional values (e.g. can be set once and referenced
      elsewhere `cfg.model.num_layer = cfg.ref.aux.num_layers`)
    trainstep: Training loop step. Do not set this field unless you need a
      custom training step.
    xm_job: XManager runtime parameters (e.g. which target is the config using)
    raw_cfg: Original config from which this `Config` was created. Automatically
      set during `konfig.resolve()`
  """

  seed: int = 0
  # usually set by the launcher
  workdir: edc.AutoCast[epath.Path] = epath.Path()
  # TODO(epot): Replace by non-TF generic protocol
  train_ds: data.Pipeline
  eval_ds: Optional[data.Pipeline] = None
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

  evals: Mapping[str, evaluators.EvaluatorBase] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  flatboards: Mapping[str, flatboard.DashboardFactory] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )
  profiler: profile_utils.Profiler = dataclasses.field(
      default_factory=profile_utils.Profiler
  )

  aux: Mapping[str, Any] = dataclasses.field(
      default_factory=flax.core.FrozenDict
  )

  trainstep: train_step.TrainStep = dataclasses.field(
      default_factory=train_step.TrainStep, repr=False
  )

  # XManager parameters
  # This is only resolved in the XManager launcher, so to avoid depending on
  # the full XManager binary, we keep the `ConfigDict` even after
  # `trainer = konfig.resolve(cfg)`
  xm_job: _JobConfigDict = dataclasses.field(default_factory=job_lib.Job)

  raw_cfg: Optional[konfig.ConfigDict] = dataclasses.field(
      default=None, repr=False
  )

  def __post_init__(self):
    # It's convenient to set `cfg.evals = None` to disable evaluation
    for name, default_factory in {
        'evals': flax.core.FrozenDict,
        'checkpointer': checkpointer_lib.NoopCheckpointer,
        'profiler': profile_utils.NoopProfiler,
    }.items():
      if getattr(self, name) is None:
        object.__setattr__(self, name, default_factory())

    # Some config object values are lazy-initialized from the root config.
    # See `UpdateFromRootCfg` for details
    for attr_name in (
        'train_ds',
        'rng_streams',
        'checkpointer',
        'profiler',
        'trainstep',
    ):
      if hasattr(self, attr_name):
        object.__setattr__(
            self, attr_name, getattr(self, attr_name).update_from_root_cfg(self)
        )
    if self.evals:
      evals = evaluators.normalize_evaluators(self.evals)
      object.__setattr__(
          self,
          'evals',
          {k: v.update_from_root_cfg(self) for k, v in evals.items()},
      )

  __konfig_resolve_exclude_fields__ = ('xm_job',)

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
    state = trainer.init_state()

    for batch in trainer.train_ds:
      batch = trainer.trainstep.step(batch, state)
    ```

    Returns:
      Final model state
      Auxiliaries
    """
    return train_lib.train_impl(self)

  @functools.cached_property
  def context_specs(self) -> context_lib.Context:
    """Shape evaluate the model (fast) and return the context structure."""
    elem_spec = self.train_ds.element_spec
    rngs = self.rng_streams.init_rngs()
    mwa = self.trainstep.model_with_aux
    state_spec = jax.eval_shape(self.init_state)

    m_batch = data_utils.mock_batch_from_elem_spec(elem_spec)
    _, context = jax.eval_shape(
        functools.partial(mwa.forward, is_training=True),
        params=state_spec.params,
        batch=m_batch,
        rngs=rngs,
        step=0,
    )
    context = context.replace(opt_state=state_spec.opt_state)
    return context
