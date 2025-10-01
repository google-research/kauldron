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

"""Evaluator that can be checkpointed."""

from __future__ import annotations

import dataclasses
import functools
import typing

import flax
import jax
from jax.experimental import checkify

from kauldron import checkpoints
# from kauldron import data
from kauldron.data import iterators
from kauldron.evals import evaluators
from kauldron.train import auxiliaries
from kauldron.train import train_step
from kauldron.utils import utils
from kauldron.utils.sharding_utils import sharding as sharding_lib  # pylint: disable=g-importing-member


@flax.struct.dataclass
class EvalState(checkpoints.items.StandardCheckpointItem):
  """State of the CheckpointedEvaluator."""

  merged_aux: auxiliaries.AuxiliariesState
  step_nr: int


class _EvalCheckpointerState(typing.NamedTuple):
  """State of the CheckpointedEvaluator."""

  eval_state: EvalState
  ds_iter: typing.Iterator[typing.Any]
  # `eval_state` is saved as the default name
  DEFAULT_ITEM = "eval_state"


# this is necessary because NamedTuple does not support double inheritance.
class EvalCheckpointerState(
    _EvalCheckpointerState, checkpoints.items.TopLevelCheckpointItem
):
  """State of the CheckpointedEvaluator."""

  pass


class CheckpointedEvaluator(evaluators.Evaluator):
  """An evaluator that can save and restore its progress."""

  checkpointer: checkpoints.checkpointer.BaseCheckpointer = dataclasses.field(
      default_factory=checkpoints.NoopCheckpointer
  )

  @functools.cached_property
  def ds_iter(self) -> iterators.Iterator:
    """Simplified iterator for checkpointing."""
    # TODO(spapa): This is a hack to get the iterator to be checkpointed. We
    # should use a proper dataset class that inherits from
    # `data.IterableDataset` and implements `__iter__` and `__len__` correctly.
    if len(self.ds) == 0:  # pylint: disable=g-explicit-length-test
      raise ValueError(
          f"Dataset for eval {self.name!r} did not yield any elements."
      )
    return self.ds

  def evaluate(
      self, state: train_step.TrainState, step: int
  ) -> auxiliaries.AuxiliariesState:
    """Run one full evaluation with checkpointing."""
    self._assert_root_cfg_resolved()
    if self.discard_opt:
      state = state.replace(opt_state=None)
    state = self.init_transform.transform(state)

    # MARK: Load state
    batch = next(iter(self.ds_iter))
    batch = sharding_lib.device_put(batch, self.base_cfg.sharding.ds)
    step_nr = 0
    step_nr_jax = sharding_lib.device_put(step_nr, sharding_lib.REPLICATED)
    aux_state = self.step(
        step_nr=step_nr_jax, state=state, batch=batch
    ).finalize()
    ds_iter = iter(self.ds_iter)

    initial_eval_state = EvalCheckpointerState(
        eval_state=EvalState(merged_aux=aux_state, step_nr=step_nr),
        ds_iter=ds_iter,
    )

    (eval_state, ds_iter) = self.checkpointer.restore(
        initial_eval_state,
        noop_if_missing=True,
    )

    latest_eval_step = eval_state.step_nr
    merged_aux = eval_state.merged_aux

    if latest_eval_step == 0:
      merged_aux = None

    # steps are 1-indexed.
    total_steps = len(self.ds) + 1

    # MARK: Run evaluation.
    for step_nr, batch in utils.enum_iter(
        ds_iter,
        init_step=latest_eval_step + 1,
        total_steps=total_steps,
        desc=self.name,
    ):
      if self.num_batches is not None:
        if step_nr > self.num_batches:
          break
      step_nr_jax = sharding_lib.device_put(step_nr, sharding_lib.REPLICATED)
      batch = sharding_lib.device_put(batch, self.base_cfg.sharding.ds)
      aux_state = self.step(
          step_nr=step_nr_jax,
          state=state,
          batch=batch,
      )

      with jax.transfer_guard("allow"):
        if aux_state.error is not None:
          checkify.check_error(aux_state.error)
        # Merge/accumulate all states
        merged_aux = merged_aux | aux_state

        # Save checkpoint.
        if self.checkpointer.should_save(step_nr):
          state_to_save = EvalCheckpointerState(
              eval_state=EvalState(
                  merged_aux=merged_aux.finalize(),
                  step_nr=step_nr,
              ),
              ds_iter=ds_iter,
          )
          self.checkpointer.save(
              state_to_save,
              step=step_nr,
          )

    if merged_aux is None:
      raise ValueError(
          f"Dataset for eval {self.name!r} did not yield any elements."
      )

    self.writer.write_step_metrics(
        step=step,
        aux=merged_aux,
        schedules={},
        log_summaries=True,
    )

    # Wait for the last checkpoint to be saved before completing the evaluation.
    self.checkpointer.wait_until_finished()
    return merged_aux
