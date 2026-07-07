# Copyright 2026 The kauldron Authors.
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
import logging
import typing
from typing import TypeVar

from etils import epath
import flax
import jax
from jax.experimental import checkify
from kauldron import checkpoints
# from kauldron import data
from kauldron.evals import evaluators
from kauldron.train import auxiliaries
from kauldron.train import train_step
from kauldron.train import trainer_lib
from kauldron.utils import utils
from kauldron.utils.sharding_utils import sharding as sharding_lib  # pylint: disable=g-importing-member
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

_SelfT = TypeVar("_SelfT")


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

  checkpointer: checkpoints.checkpointer.Checkpointer
  log_tqdm_xm: bool = False

  def __post_init__(self):
    super().__post_init__()
    if isinstance(self.base_cfg, trainer_lib.Trainer):
      # check if self.checkpointer exists, has attr workdir, and if the workdir
      # is None, then set it to the eval workdir.
      if hasattr(self, "checkpointer") and hasattr(
          self.checkpointer, "workdir"
      ):
        workdir = epath.Path(self.base_cfg.workdir)
        # check the workdir for the checkpointer has not been changed.
        if self.checkpointer.workdir == workdir:
          eval_workdir = workdir / "evals" / self.name
          new_checkpointer = dataclasses.replace(
              self.checkpointer, workdir=eval_workdir
          )
          object.__setattr__(self, "checkpointer", new_checkpointer)
    if self.cache:
      raise TypeError("CheckpointedEvaluator does not support caching, yet.")

  def _get_init_aux_state(
      self, state: train_step.TrainState
  ) -> auxiliaries.AuxiliariesState:
    """Get the initial aux state from the first batch."""
    batch = next(iter(self.ds))
    batch = sharding_lib.device_put(batch, self.base_cfg.sharding.batch)
    step_nr_jax = sharding_lib.device_put(0, sharding_lib.REPLICATED)
    return self.step(step_nr=step_nr_jax, state=state, batch=batch).finalize()

  def _eval_done_path(self, step: int) -> epath.Path:
    """Return the path to the marker file indicating eval is done."""
    return epath.Path(self.checkpointer.workdir) / f"step_{step}_done"

  def _step_checkpointer(
      self, step: int
  ) -> checkpoints.checkpointer.Checkpointer:
    """Create a checkpointer scoped to a specific training step.

    This prevents checkpoint clashes when the evaluator is called at multiple
    training steps (e.g. step 1000, step 2000).

    Args:
      step: The training step to scope the checkpointer to.

    Returns:
      A checkpointer scoped to the training step.
    """
    step_workdir = epath.Path(self.checkpointer.workdir) / f"step_{step}"
    return dataclasses.replace(self.checkpointer, workdir=step_workdir)

  def evaluate(
      self, state: train_step.TrainState, step: int
  ) -> auxiliaries.AuxiliariesState:
    """Run one full evaluation with checkpointing."""
    self._assert_root_cfg_resolved()

    # Skip evaluation if it already completed (handles preemption restarts).
    done_path = self._eval_done_path(step)
    if done_path.exists():
      logging.info(
          "Eval %r at step %d already completed. Skipping.",
          self.name,
          step,
      )
      return auxiliaries.AuxiliariesState()

    if self.discard_opt:
      state = state.replace(opt_state=None)
    state = self.init_transform.transform(state)

    # Use a step-specific checkpointer so that eval runs at different training
    # steps don't overwrite each other's checkpoints.
    ckptr = self._step_checkpointer(step)

    # MARK: Load state
    step_nr = 0
    aux_state = self._get_init_aux_state(state)
    ds_iter = iter(self.ds)

    initial_eval_state = EvalCheckpointerState(
        eval_state=EvalState(merged_aux=aux_state, step_nr=step_nr),
        ds_iter=ds_iter,
    )

    eval_state, ds_iter = ckptr.restore(
        initial_eval_state,
        noop_if_missing=True,
    )

    latest_eval_step = eval_state.step_nr
    merged_aux = eval_state.merged_aux

    if latest_eval_step == 0:
      merged_aux = None

    # steps are 1-indexed.
    try:
      total_steps = len(self.ds) + 1
    except TypeError:  # Unknown length.
      total_steps = None

    if self.num_batches is not None:
      if total_steps is None:
        total_steps = self.num_batches + 1
      else:
        total_steps = min(total_steps, self.num_batches + 1)

    # MARK: Run evaluation.
    for step_nr, batch in utils.enum_iter(
        ds_iter,
        init_step=latest_eval_step + 1,
        total_steps=total_steps,
        desc=self.name,
        log_xm=self.log_tqdm_xm,
    ):
      if self.num_batches is not None:
        if step_nr > self.num_batches:
          break
      step_nr_jax = sharding_lib.device_put(step_nr, sharding_lib.REPLICATED)
      batch = sharding_lib.device_put(batch, self.base_cfg.sharding.batch)
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
        if ckptr.should_save(step_nr):
          state_to_save = EvalCheckpointerState(
              eval_state=EvalState(
                  merged_aux=merged_aux.finalize(),
                  step_nr=step_nr,
              ),
              ds_iter=ds_iter,
          )
          ckptr.save(
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
    ckptr.wait_until_finished()
    # Clean up step-specific checkpoints after successful evaluation, since
    # they are only needed for crash recovery during the eval run.
    if status.is_lead_host:
      step_workdir = epath.Path(ckptr.workdir)
      if step_workdir.exists():
        step_workdir.rmtree()
      # Write a marker file to indicate eval completed successfully.
      done_path.touch()
    return merged_aux
