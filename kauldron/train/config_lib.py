# Copyright 2024 The kauldron Authors.
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

from collections.abc import MutableMapping
import dataclasses
import functools
import typing
from typing import Any, Optional

from etils import edc
from etils import epath
import flax
from flax import linen as nn
import jax
from jax.experimental import checkify
from kauldron import checkpoints
from kauldron import data
from kauldron import konfig
from kauldron import losses
from kauldron import metrics
from kauldron import summaries
from kauldron.data import utils as data_utils
from kauldron.evals import eval_impl
from kauldron.evals import evaluators
from kauldron.inspect import profile_utils
from kauldron.train import metric_writer
from kauldron.train import rngs_lib
from kauldron.train import setup_utils
from kauldron.train import train_lib
from kauldron.train import train_step
from kauldron.utils import config_util
from kauldron.utils import context as context_lib
from kauldron.utils.sharding_utils import sharding as sharding_utils  # pylint: disable=g-importing-member
import optax

with konfig.imports(lazy=True):
  # Do not resolve job_lib to not link the full XManager API to Kauldron
  from kauldron.xm._src import job_lib  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


# TODO(epot): There's some strange interaction between `get_type_hints` from
# `edc`, `ConfigDictLike` and lazy job imports. Should investigate but as it's
# not critical, just use this workaround.
if typing.TYPE_CHECKING:
  # `kxm.Job` is only resolved in the XManager launcher, to avoid depending on
  # the full XManager binary, we keep the `ConfigDict` even after
  # `trainer = konfig.resolve(cfg)`
  # This is specified through `__konfig_resolve_exclude_fields__`
  _JobConfigDict = konfig.ConfigDictLike[job_lib.Job]
else:
  _JobConfigDict = Any

# This hack is needed because `JaxException` is used to define ErrorCategory
# as Type['JaxException'] before class JaxException is declared.
CheckifyErrorCategory = checkify.ErrorCategory if typing.TYPE_CHECKING else Any

FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict


class Trainer(config_util.BaseConfig):
  """Base trainer class.

  This class is the root object containing all the configuration options (
  datasets, model, optimizer, etc.).

  Usage:

  ```python
  trainer = kd.train.Trainer(
      train_ds=...,
      model=...,
      optimizer=...,
      ...
  )
  trainer.train()
  ```

  Attributes:
    seed: Seed for all rngs
    workdir: Root dir of the experiment (usually set by XManager)
    train_ds: Dataset used in training
    eval_ds: Dataset used in eval (see https://kauldron.rtfd.io/en/latest/eval.html to activate eval)
    model: Flax linen module
    rng_streams: Flax rng streams to use **in addition** of the default
      (`params`, `dropout`, `default`). If any of `params`, `dropout`, `default`
      is set here, it will overwrite the default value.
    sharding: Model sharding (by default, use replicated sharding)
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
    writer: Metric writer used for writing to TB, datatable, etc.
    profiler: Profiler can be customized (see `kd.inspect.Profile`)
    checkify_error_categories: List of errors to enable checkify for.
    schedules: optax schedules (to be used in `optimizer`)
    optimizer: optax optimizer
    checkpointer: Checkpoint used to save/restore the state
    init_transforms: A `dict` for initial state transformation. Used for partial
      checkpoint loading (re-use pre-trained weights).
    trainstep: Training loop step. Do not set this field unless you need a
      custom training step.
    evals: Evaluators to use (e.g. `{'eval': kd.eval.Evaluator()}`)
    aux: Arbitrary additional values (e.g. can be set once and referenced
      elsewhere `cfg.model.num_layer = cfg.ref.aux.num_layers`)
    setup: Global setup options
    xm_job: XManager runtime parameters (e.g. which target is the config using)
    raw_cfg: Original config from which this `Config` was created. Automatically
      set during `konfig.resolve()`
  """

  seed: int = 0
  # usually set by the launcher
  workdir: edc.AutoCast[epath.Path] = epath.Path()

  # Data pipeline
  train_ds: data.Pipeline
  eval_ds: Optional[data.Pipeline] = None

  # Model
  model: nn.Module
  rng_streams: rngs_lib.RngStreams = dataclasses.field(
      default_factory=rngs_lib.RngStreams
  )
  sharding: sharding_utils.ShardingStrategy = dataclasses.field(
      default_factory=sharding_utils.ShardingStrategy
  )
  num_train_steps: Optional[int] = None
  stop_after_steps: Optional[int] = None

  # Metrics, losses, summaries
  log_metrics_every: int = 100
  log_summaries_every: int = 1000
  train_losses: MutableMapping[str, losses.Loss] = dataclasses.field(
      default_factory=FrozenDict
  )
  train_metrics: MutableMapping[str, metrics.Metric] = dataclasses.field(
      default_factory=FrozenDict
  )
  train_summaries: MutableMapping[str, summaries.Summary] = dataclasses.field(
      default_factory=FrozenDict
  )

  writer: metric_writer.WriterBase = dataclasses.field(
      default_factory=metric_writer.KDMetricWriter
  )
  profiler: profile_utils.Profiler = dataclasses.field(
      default_factory=profile_utils.Profiler
  )
  checkify_error_categories: frozenset[CheckifyErrorCategory] = frozenset()

  # Optimizer
  schedules: MutableMapping[str, optax.Schedule] = dataclasses.field(
      default_factory=FrozenDict
  )
  optimizer: optax.GradientTransformation

  # Checkpoints
  checkpointer: checkpoints.checkpointer.BaseCheckpointer = dataclasses.field(
      default_factory=checkpoints.NoopCheckpointer
  )
  init_transforms: MutableMapping[str, checkpoints.AbstractPartialLoader] = (
      dataclasses.field(default_factory=FrozenDict)
  )

  # Train, eval loop
  trainstep: train_step.TrainStep = dataclasses.field(
      default_factory=train_step.TrainStep, repr=False
  )
  evals: MutableMapping[str, evaluators.EvaluatorBase] = dataclasses.field(
      default_factory=FrozenDict
  )

  # Additional arbitrary config values
  # Should this be renamed `extra` ?
  aux: MutableMapping[str, Any] = dataclasses.field(default_factory=FrozenDict)

  # XManager and other environement parameters
  setup: setup_utils.Setup = dataclasses.field(
      default_factory=setup_utils.Setup
  )
  xm_job: _JobConfigDict = dataclasses.field(default_factory=job_lib.Job)

  # Original `konfig.ConfigDict` from which the `Trainer` was created.
  raw_cfg: Optional[konfig.ConfigDict] = dataclasses.field(
      default=None, repr=False
  )

  def __post_init__(self):
    # It's convenient to set `cfg.evals = None` to disable evaluation
    for name, default_factory in {
        'evals': FrozenDict,
        'checkpointer': checkpoints.NoopCheckpointer,
        'profiler': profile_utils.NoopProfiler,
        'writer': metric_writer.NoopWriter,
    }.items():
      if getattr(self, name) is None:
        object.__setattr__(self, name, default_factory())

    # TODO(epot):
    # Use self.update_from_root_cfg(self) directly

    # Some config object values are lazy-initialized from the root config.
    # See `UpdateFromRootCfg` for details
    for attr_name in (
        'train_ds',
        'eval_ds',
        'rng_streams',
        'checkpointer',
        'profiler',
        'trainstep',
        'writer',
    ):
      if (value := getattr(self, attr_name)) is not None:
        object.__setattr__(self, attr_name, value.update_from_root_cfg(self))
    if self.evals:
      evals = evaluators.normalize_evaluators(self.evals)
      object.__setattr__(
          self,
          'evals',
          {k: v.update_from_root_cfg(self) for k, v in evals.items()},
      )

    # set the name of the collection to train
    if isinstance(self.writer, metric_writer.KDMetricWriter):
      object.__setattr__(
          self,
          'writer',
          dataclasses.replace(self.writer, collection='train'),
      )

  __konfig_resolve_exclude_fields__ = ('xm_job',)

  def __post_konfig_resolve__(self, cfg: konfig.ConfigDict) -> None:
    """Bind the raw config to kd. Called during `kd.konfig.resolve()`."""
    # TODO(epot): Should freeze and deep-copy the config, but let's do this
    # after fiddle migration.
    object.__setattr__(self, 'raw_cfg', cfg)

  # Do not use property to make it explicit this is recomputed each time
  def init_state(
      self, *, skip_transforms: bool = False
  ) -> train_step.TrainState:
    """Create the state: `cfg.trainstep.init(cfg.train_ds.element_spec)`."""
    return self.trainstep.init(
        self.train_ds.element_spec, skip_transforms=skip_transforms
    )

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

  def continuous_eval(
      self,
      names: str | list[str],
      final_eval_names: str | list[str] | None = None,
  ) -> dict[str, train_step.Auxiliaries]:
    """Main method that perform auxiliary tasks (evaluation, rendering,...).

    Trigger an evaluation everytime a new checkpoint is detected.

    See https://kauldron.rtfd.io/en/latest/eval.html for details.

    Args:
      names: Name of the evaluators to run.
      final_eval_names: Name of the evaluators to run at the end of the
        training.

    Returns:
      Auxiliaries: Mapping eval name to auxiliary
    """
    if isinstance(names, str):
      names = [names]
    if final_eval_names is None:
      final_eval_names = []
    if isinstance(final_eval_names, str):
      final_eval_names = [final_eval_names]
    return eval_impl.continuous_eval(
        self,
        eval_names=names,
        final_eval_names=final_eval_names,
    )

  @functools.cached_property
  def state_specs(self) -> train_step.TrainState:
    """Returns the `state` specs."""
    # Cache `element_spec` before calling the jitted function, otherwise
    # `rng` used inside the data pipelines are mocked with Traced<Array>
    _ = self.train_ds.element_spec

    # Skip the `init_transforms`. Indeed, restoring checkpoint (partial
    # loading) will fail inside `jax.eval_shape / `jax.jit`
    init_fn = functools.partial(self.init_state, skip_transforms=True)
    return jax.eval_shape(init_fn)

  @functools.cached_property
  def context_specs(self) -> context_lib.Context:
    """Shape evaluate the model (fast) and return the context structure."""
    elem_spec = self.train_ds.element_spec
    elem_sharding = self.sharding.ds
    rngs = self.rng_streams.init_rngs()
    mwa = self.trainstep.model_with_aux

    state_specs = self.state_specs

    m_batch = data_utils.mock_batch_from_elem_spec(elem_spec, elem_sharding)
    _, context = jax.eval_shape(
        functools.partial(mwa.forward, is_training=True),
        params=state_specs.params,
        batch=m_batch,
        rngs=rngs,
        step=0,
        collections=state_specs.collections,
    )
    context = context.replace(opt_state=state_specs.opt_state)
    return context
