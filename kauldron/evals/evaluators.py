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

"""Evaluators."""

from __future__ import annotations

import collections.abc
import dataclasses
import functools
import typing
from typing import Any, Optional, TypeVar

from etils import epy
from flax import linen as nn
import jax
from jax.experimental import checkify
from kauldron import checkpoints
from kauldron import data
from kauldron import losses as losses_lib
from kauldron import metrics as metrics_lib
from kauldron.evals import run_strategies
from kauldron.train import auxiliaries
from kauldron.train import context as context_lib
from kauldron.train import metric_writer
from kauldron.train import rngs_lib
from kauldron.train import train_step
from kauldron.train import trainer_lib
from kauldron.utils import config_util
from kauldron.utils import immutabledict
from kauldron.utils import kdash
from kauldron.utils import utils
from kauldron.utils.sharding_utils import sharding as sharding_lib  # pylint: disable=g-importing-member
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

_SelfT = TypeVar('_SelfT')

_DEFAULT_EVAL_NAME = 'eval'

CheckifyErrorCategory = checkify.ErrorCategory if typing.TYPE_CHECKING else Any


@dataclasses.dataclass(kw_only=True, frozen=True)
class CollectionKeys:
  """Names of the metrics/summaries/losses (displayed in flatboard)."""

  losses: tuple[str, ...] = ()
  metrics: tuple[str, ...] = ()
  summaries: tuple[str, ...] = ()


class EvaluatorBase(config_util.BaseConfig, config_util.UpdateFromRootCfg):
  """Base class for inline evaluators.

  Evaluators should inherit from this class and implement the `evaluate` method.

  Attributes:
    name: Eval name (collection name for TensorBoard and Datatables)
    run: How/when to run this eval (e.g. `kd.evals.EveryNSteps(100)` or
      `kd.evals.StandaloneEveryCheckpoint()`)
    writer: Metric writer (set automatically)
    base_cfg: reference to the experiment configuration (set automatically).
    discard_opt: Whether to discard the optimizer state for the evaluator. This
      is useful to save memory in case the evaluator does not need access to the
      optimizer state.
  """

  # Evaluators can be used as standalone, so keep a default name
  name: str = _DEFAULT_EVAL_NAME

  run: run_strategies.RunStrategy

  writer: metric_writer.WriterBase = dataclasses.field(
      default=config_util.ROOT_CFG_REF.writer
  )
  base_cfg: trainer_lib.Trainer = dataclasses.field(
      default=config_util.ROOT_CFG_REF, repr=False
  )

  discard_opt: bool = False

  def __post_init__(self) -> None:
    if hasattr(super(), '__post_init__'):
      super().__post_init__()  # Future proof to run `__post_init__` in parents  # pylint: disable=attribute-error
    if not self.name.replace('.', '_').replace('-', '_').isidentifier():
      raise ValueError(
          'Evaluator name should be a valid Python identifier. Got:'
          f' {self.name}'
      )

    # always set the name of the collection to eval name
    if isinstance(self.writer, metric_writer.WriterBase):
      object.__setattr__(
          self,
          'writer',
          dataclasses.replace(self.writer, collection=self.name),
      )

  def maybe_eval(self, *, step: int, state: train_step.TrainState) -> Any:
    """Run or skip the evaluator for the given train-step."""
    if self.run.should_eval_in_train(step):
      try:
        return self.evaluate(state, step)
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Chain exception so we see the name of the Evaluator that failed.
        epy.reraise(e, f'Evaluator {self.name} failed at step {step}: ')

  def evaluate(self, state: train_step.TrainState, step: int) -> Any:
    """Run this evaluator then write and optionally return the results."""
    raise NotImplementedError

  @functools.cached_property
  def __dashboards__(self) -> kdash.DashboardsBase:
    """Returns collection keys used by flatboard.

    To be merged with the other existing metrics.

    Returns:
      A dict mapping collection keys (usually `self.name` with the metrics,
      summaries,... names).
    """
    # TODO(epot): Make this abstract and update childs
    status.log(
        f'Warning: Evaluator {type(self).__name__} ({self.name!r}) does not'
        ' implement `__dashboards__` protocol, so NO metrics will be reported'
        ' to flatboard. This will be an error in the future.\nSee'
        ' `kd.evals.Evaluator.__dashboards__` for an example (returning'
        ' `kdash.MetricDashboards`). If your evaluator does not report any'
        ' metrics, you can return an empty `kdash.NoopDashboards()`'
    )
    return kdash.NoopDashboard()


class Evaluator(EvaluatorBase):
  """Evaluator running `num_batches` times.

  Evaluators can be launched as separate XManager jobs
  (e.g. `run=kd.evals.StandaloneEveryCheckpoint()`) or along train
  (e.g. `run=kd.evals.EveryNSteps(100)`).

  If not provided, losses, metrics, summaries are reused from train.

  Usage:

  ```
  evaluator = kd.evals.Evaluator(
      run=kd.evals.EveryNSteps(100),
      ds=test_ds,
      base_cfg=cfg,
  )
  evaluator.maybe_eval(step=0, state=state)
  ```

  Attributes:
    num_batches: How many batches to run evaluation on. Use `None` to evaluate
      on the full test dataset. Note that each evaluation reinitializes the
      dataset iterator, so setting to `1` will run all evaluations on the same
      batch.
    cache: Whether to cache the iterator
    ds: Dataset to evaluate on.
    losses: Losses
    metrics: Metrics
    summaries: Summaries
    model: Model to use for evaluation (if different from train).
    model_method: Name of the flax model method to use (defaults to `__call__`)
    init_transform: Transform to apply to the state before evaluation. This is
      useful for example for replacing the weights of the network with EMA
      weights.
  """

  num_batches: Optional[int] = None
  cache: bool = False
  ds: data.Pipeline = config_util.ROOT_CFG_REF.eval_ds
  losses: dict[str, losses_lib.Loss] = config_util.ROOT_CFG_REF.train_losses
  metrics: dict[str, metrics_lib.Metric] = (
      config_util.ROOT_CFG_REF.train_metrics
  )
  summaries: dict[str, metrics_lib.Metric] = (
      config_util.ROOT_CFG_REF.train_summaries
  )
  model: nn.Module = config_util.ROOT_CFG_REF.model
  model_method: Optional[str] = None
  init_transform: checkpoints.AbstractPartialLoader = dataclasses.field(
      default_factory=lambda: checkpoints.NoopTransform(),  # pylint: disable=unnecessary-lambda
  )
  checkify_error_categories: frozenset[CheckifyErrorCategory] = (
      config_util.ROOT_CFG_REF.checkify_error_categories
  )

  def __post_init__(self) -> None:
    super().__post_init__()

    immutabledict.freeze_dict_attrs(self, ['losses', 'metrics', 'summaries'])

    if self.ds is None:
      raise ValueError(
          f'Eval dataset missing (`cfg.evals.{self.name}.ds is None`). Please'
          ' set it either in `kd.train.Trainer.eval_ds` or in'
          ' `Evaluator(ds=...)`.'
      )

  @functools.cached_property
  def ds_iter(self) -> data.IterableDataset:
    """"""
    ds_iter = self.ds
    if self.num_batches is not None:
      ds_iter = ds_iter.take(self.num_batches)
    if self.cache:
      if self.num_batches is None:
        raise ValueError('Can only cache if num_batches is set.')
      ds_iter = ds_iter.cache()
    return ds_iter.device_put(self.base_cfg.sharding.ds)

  @functools.cached_property
  def aux(self) -> auxiliaries.Auxiliaries:
    return auxiliaries.Auxiliaries(
        losses=self.losses, metrics=self.metrics, summaries=self.summaries
    )

  def evaluate(
      self, state: train_step.TrainState, step: int
  ) -> auxiliaries.AuxiliariesState:
    """Run one full evaluation."""
    self._assert_root_cfg_resolved()
    if self.discard_opt:
      state = state.replace(opt_state=None)
    state = self.init_transform.transform(state)

    # TODO(epot): Add chrono to evals. Note: One issue is that the
    # write metric time will be excluded from the chrono (including the final).
    # metric computation time. Is there a better way ?

    merged_aux = None
    for step_nr, batch in utils.enum_iter(self.ds_iter, desc=self.name):
      step_nr = sharding_lib.device_put(step_nr, sharding_lib.REPLICATED)
      aux_state = self.step(
          step_nr=step_nr,
          state=state,
          batch=batch,
      )
      # Raise any checkify errors that were encountered during step.
      if aux_state.error is not None:
        checkify.check_error(aux_state.error)
      # Merge/accumulate all states
      with jax.transfer_guard('allow'):
        merged_aux = merged_aux | aux_state
    if merged_aux is None:  # At least one iteration
      raise ValueError(
          f'Dataset for eval {self.name!r} did not yield any elements:\n'
          f'{epy.pretty_repr(self.ds)}'
      )

    self.writer.write_step_metrics(
        step=step,
        aux=merged_aux,
        schedules={},
        log_summaries=True,
    )
    return merged_aux

  @functools.partial(
      jax.jit,
      static_argnames=('self',),
  )
  def step(
      self,
      *,
      step_nr: int,
      state: train_step.TrainState,
      batch: Any,
  ) -> auxiliaries.AuxiliariesState:
    with self.base_cfg.sharding.set_global_mesh():
      if self.checkify_error_categories:
        step_fn = checkify.checkify(
            self._step, errors=self.checkify_error_categories
        )
        error, aux_state = step_fn(step_nr, state, batch)
      else:
        error = None
        aux_state = self._step(step_nr, state, batch)

    aux_state = aux_state.replace(error=error)

    return sharding_lib.with_sharding_constraint(
        aux_state, self.base_cfg.sharding.aux
    )

  def _step(
      self, step_nr: int, state: train_step.TrainState, batch: Any
  ) -> auxiliaries.AuxiliariesState:
    """Eval step to be wrapped by checkify and called by `step`.

    Subclasses can overwrite this method to implement custom evaluation
    steps.

    Args:
      step_nr: The evaluation step number.
      state: The current training state.
      batch: The batch to use for the evaluation step.

    Returns:
      The auxiliary state with the evaluation results for this batch.
    """
    # Note that ctx.step is train step (from train state), NOT `eval_step`
    ctx = context_lib.Context.from_state_and_batch(state=state, batch=batch)
    ctx = train_step.forward(
        model=self.model,
        context=ctx,
        rngs=self.base_cfg.rng_streams.eval_rngs(step_nr),
        is_training=False,
        method=self.model_method,
    )
    ctx = self.aux.update_context(ctx)
    return ctx.get_aux_state(
        return_losses=True, return_metrics=True, return_summaries=True
    )

  @functools.cached_property
  def model_with_aux(self) -> train_step.ModelWithAux:
    """Deprecated. Use a forward function directly instead.

    See e.g. `kd.train.train_step.forward_with_loss`.
    """
    return train_step.ModelWithAux(
        model=self.model,
        losses=self.losses,
        metrics=self.metrics,
        summaries=self.summaries,
    )

  @functools.cached_property
  def __dashboards__(self) -> kdash.DashboardsBase:
    return kdash.MetricDashboards(
        collection=self.name,
        losses=self.losses,
        metrics=self.metrics,
    )

  def __hash__(self) -> int:
    # Make Evaluator hashable, so its methods can be jitted.
    return id(self)


@utils.checkify_ignore
@functools.partial(
    jax.jit,
    static_argnames=(
        'model',
        'rng_streams',
        'aux',
        'sharding',
        'model_method',
    ),
)
def basic_eval_step(
    *,
    model: nn.Module,
    rng_streams: rngs_lib.RngStreams,
    aux: auxiliaries.Auxiliaries,
    eval_step: int,
    state: train_step.TrainState,
    batch,
    sharding: sharding_lib.ShardingStrategy,
    model_method: str | None = None,
) -> auxiliaries.AuxiliariesState:
  """DEPRECATED. Use / override `Evaluator._step` instead."""
  # Note that ctx.step is train step (from train state), NOT `eval_step`
  ctx = context_lib.Context.from_state_and_batch(state=state, batch=batch)

  with sharding.set_global_mesh():
    ctx = train_step.forward(
        model=model,
        context=ctx,
        rngs=rng_streams.eval_rngs(eval_step),
        is_training=False,
        method=model_method,
    )

  ctx = aux.update_context(ctx)
  aux_state = ctx.get_aux_state(
      return_losses=True, return_metrics=True, return_summaries=True
  )

  return sharding_lib.with_sharding_constraint(aux_state, sharding.aux)


def normalize_evaluators(
    evaluators: collections.abc.Mapping[str, EvaluatorBase],
) -> collections.abc.Mapping[str, EvaluatorBase]:
  """Set the evaluator names."""
  if not isinstance(evaluators, collections.abc.Mapping):
    raise TypeError(
        '`cfg.evals` should be a `dict[str, EvaluatorBase]`. Got:'
        f' {type(evaluators)}'
    )
  return {k: _replace_name(c, k) for k, c in evaluators.items()}


def _replace_name(evaluator: EvaluatorBase, name: str) -> EvaluatorBase:
  """Set the `evaluator.name`."""
  if not isinstance(evaluator, EvaluatorBase):
    raise TypeError(
        'Eval values should be `kd.evals.EvaluatorBase`. Got:'
        f' {name}={type(evaluator)}'
    )
  elif name == 'train':
    raise ValueError(
        'Evaluator cannot be named `train` as it conflict with training'
        ' metrics.'
    )
  elif evaluator.name == _DEFAULT_EVAL_NAME:  # Default name, overwrite
    return dataclasses.replace(evaluator, name=name)
  elif evaluator.name == name:
    return evaluator
  else:
    raise ValueError(
        f'Evaluator name provided should match. Got: {evaluator.name} != {name}'
    )
