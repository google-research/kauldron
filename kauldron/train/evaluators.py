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

"""Evaluators."""

from __future__ import annotations

import abc
import dataclasses
import functools
import itertools
from typing import Any, Optional, TypeVar

import flax
import jax
from kauldron import data
from kauldron import losses as losses_lib
from kauldron import metrics as metrics_lib
from kauldron import summaries as summaries_lib
from kauldron.train import config_lib
from kauldron.train import metric_writer
from kauldron.train import rngs_lib
from kauldron.train import train_lib
from kauldron.train import train_step
from kauldron.utils import config_util
from kauldron.utils import utils

_SelfT = TypeVar('_SelfT')

_REUSE_TRAIN: Any = object()


class EvaluatorBase(
    config_util.BaseConfig, config_util.UpdateFromRootCfg, abc.ABC
):
  """Evaluator interface.

  Usage:

  ```
  evaluator = SimpleEvaluator(
      run_every=100,
      ds=test_ds,
      base_cfg=cfg,
  )
  evaluator.maybe_eval(step=0, state=state)
  ```
  """

  @abc.abstractmethod
  def maybe_eval(self, *, step: int, state: train_step.TrainState):
    """Eventually evaluate the train state."""
    raise NotImplementedError

  @abc.abstractmethod
  def flatten(self) -> list[EvaluatorBase]:
    """Iterate over the evaluator nodes."""
    raise NotImplementedError


class NoopEvaluator(EvaluatorBase):
  """No evaluation."""

  def maybe_eval(self, *, step: int, state: train_step.TrainState) -> None:
    pass

  def flatten(self) -> list[EvaluatorBase]:
    return []


class SingleEvaluator(EvaluatorBase):
  """Evaluator running `num_batches` times every `run_every` steps.

  If not provided, losses, metrics, summaries are reused from train.

  Attributes:
    name: Eval name (display in TensorBoard)
    run_every: Run eval every `run_every` train steps
    num_batches: How many batches to run evaluation on. Use `None` to evaluate
      on the full test dataset. Note that each evaluation reinitializes the
      dataset iterator, so setting to `1` will run all evaluations on the same
      batch.
    ds: Dataset to evaluate on.
    losses: Losses
    metrics: Metrics
    summaries: Summaries
  """

  name: str = 'eval'
  run_every: int
  num_batches: Optional[int]
  ds: data.TFDataPipeline = config_util.ROOT_CFG_REF.eval_ds
  losses: dict[str, losses_lib.Loss] = config_util.ROOT_CFG_REF.train_losses
  metrics: dict[str, metrics_lib.Metric] = (
      config_util.ROOT_CFG_REF.train_metrics
  )
  summaries: dict[str, summaries_lib.Summary] = (
      config_util.ROOT_CFG_REF.train_summaries
  )

  base_cfg: config_lib.Config = dataclasses.field(
      default=config_util.ROOT_CFG_REF, repr=False
  )

  # TODO(klausg): filter out metrics / summaries that access grads/updates

  def update_from_root_cfg(self: _SelfT, root_cfg: config_lib.Config) -> _SelfT:
    """See base class."""
    new_self = super().update_from_root_cfg(root_cfg)
    if new_self.ds is None:
      raise ValueError(
          'Eval dataset missing (`SingleEvaluator.ds is None`). Please set it'
          ' either in `kd.train.Config.eval_ds` or in `SingleEvaluator.ds`.'
      )
    return new_self.replace(
        ds=new_self.ds.update_from_root_cfg(root_cfg),
    )

  def maybe_eval(
      self, *, step: int, state: train_step.TrainState
  ) -> train_step.Auxiliaries | None:
    """See base class."""
    if self.should_eval(step):
      return self.eval_step(state, step)

  def should_eval(self, step: int) -> bool:
    return step % self.run_every == 0

  def eval_step(
      self, state: train_step.TrainState, step: int
  ) -> train_step.Auxiliaries:
    """Run one full evaluation."""
    self._assert_root_cfg_resolved()

    merged_aux = None
    for eval_step, batch in utils.enum_iter(
        self.ds,
        total_steps=self.num_batches,
        desc='eval',
    ):
      eval_step = flax.jax_utils.replicate(eval_step)
      aux = _pstep(
          self.model_with_aux,
          self.base_cfg.rng_streams,
          eval_step,
          state,
          batch,
      )
      # Merge/accumulate all states
      if merged_aux is None:
        merged_aux = aux
      else:
        merged_aux = merged_aux.merge(aux)
    assert merged_aux is not None  # At least one iteration

    train_lib.write_summaries(
        writer=self.writer,
        step=step,
        aux=merged_aux,
        schedules={},
        model_with_aux=self.model_with_aux,
        log_summaries=True,
    )
    return merged_aux

  @functools.cached_property
  def model_with_aux(self) -> train_step.ModelWithAux:
    """Model which also compute the auxiliaries (losses, metrics,...)."""
    return train_step.ModelWithAux(
        model=self.base_cfg.model,
        losses=self.losses,
        metrics=self.metrics,
        summaries=self.summaries,
    )

  @functools.cached_property
  def writer(self) -> metric_writer.KDMetricWriter:
    """Metric writer."""
    return metric_writer.KDMetricWriter(
        workdir=self.base_cfg.workdir, collection=self.name
    )

  def flatten(self) -> list[EvaluatorBase]:
    return [self]


@functools.partial(
    jax.pmap,
    axis_name='device',
    static_broadcasted_argnums=(0, 1),
)
def _pstep(
    model_with_aux: train_step.ModelWithAux,
    rng_streams: rngs_lib.RngStreams,
    eval_step: int,
    state: train_step.TrainState,
    batch,
) -> train_step.Auxiliaries:
  """Call the model (pmap version)."""
  _, ctx = model_with_aux.forward(
      params=state.params,
      batch=batch,
      rngs=rng_streams.eval_rngs(
          eval_step, device_id=jax.lax.axis_index('device')
      ),
      step=state.step,  # Step is train step, NOT eval
      is_training=False,
  )
  aux = model_with_aux.get_aux(
      ctx,
      return_losses=True,
      return_metrics=True,
      return_summaries=True,
  )
  return aux


class MultiEvaluator(EvaluatorBase):
  """Evaluator which contain individual evaluators.

  Usage:

  ```
  evaluator = kd.train.MultiEvaluator(
      children=[
          kd.train.SingleEvaluator(name='eval0'),
          kd.train.SingleEvaluator(name='eval1'),
      ]
  )
  evaluator.maybe_eval(step=0, state=state)
  ```
  """

  children: list[EvaluatorBase]

  def update_from_root_cfg(self: _SelfT, root_cfg: config_lib.Config) -> _SelfT:
    """See base class."""
    return self.replace(
        children=[c.update_from_root_cfg(root_cfg) for c in self.children]
    )

  def maybe_eval(self, *, step: int, state: train_step.TrainState):
    for evaluator in self.children:
      evaluator.maybe_eval(step=step, state=state)

  def flatten(self) -> list[EvaluatorBase]:
    return list(
        itertools.chain.from_iterable(eval.flatten() for eval in self.children)
    )
