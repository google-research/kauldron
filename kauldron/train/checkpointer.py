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

"""Implementations of Orbax checkpointers for typical usecases."""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional, Sequence

from etils import epath
from flax.training import orbax_utils
import jax
import orbax.checkpoint as orbax


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class Checkpointer:
  """Basic Orbax Checkpointmanager."""
  workdir: str | epath.Path

  save_interval_steps: int
  max_to_keep: Optional[int] = None  # Keep all.

  @functools.cached_property
  def _ckpt_mgr(self) -> orbax.checkpoint_manager.CheckpointManager:
    """Returns checkpoint manager instance (initialized and cached)."""
    mgr_options = orbax.CheckpointManagerOptions(
        save_interval_steps=self.save_interval_steps,
        max_to_keep=self.max_to_keep,
        step_prefix="ckpt",
        # TODO(msajjadi): Re-enable this once we've figured it out.
        # step_format_fixed_length=9,
        create=True,
    )
    ckpt_mgr = orbax.CheckpointManager(
        epath.Path(self.workdir) / "checkpoints",
        orbax.Checkpointer(orbax.PyTreeCheckpointHandler()),
        options=mgr_options,
    )
    return ckpt_mgr

  def restore(
      self,
      initial_state,
      step: int = -1,
      *,
      noop_if_missing: bool = False,
  ):
    """Restore state."""
    state = initial_state
    if self._ckpt_mgr.latest_step() is not None:
      step = self._ckpt_mgr.latest_step() if step == -1 else step
      assert (
          step in self._ckpt_mgr.all_steps()
      ), f"No checkpoint is available for step {step}"

      target, structure = jax.tree_util.tree_flatten(initial_state)
      restore_args = orbax_utils.restore_args_from_target(target)
      state = self._ckpt_mgr.restore(
          step,
          items=target,
          restore_kwargs={"restore_args": restore_args},
      )
      state = jax.tree_util.tree_unflatten(structure, state)
    elif not noop_if_missing:  # No checkpoint
      raise ValueError(
          "Could not restore checkpoint. Use `noop_if_missing=True`"
          " to default to initial state."
      )
    return state

  def should_save(self, step: int) -> bool:
    return self._ckpt_mgr.should_save(step)

  def save_state(
      self,
      state,
      step: int,
      *,
      force: bool = False,
  ) -> bool:
    """Save state."""

    target = jax.tree_util.tree_leaves(state)
    save_args = orbax_utils.save_args_from_target(target)
    return self._ckpt_mgr.save(
        step, target, save_kwargs={"save_args": save_args}, force=force
    )

  def maybe_save_state(
      self,
      state,
      step: int,
      *,
      force: bool = False,
  ) -> bool:
    """Save state."""
    if not self.should_save(step):
      return False

    return self.save_state(state, step, force=force)

  @property
  def latest_step(self) -> Optional[int]:
    return self._ckpt_mgr.latest_step()

  @property
  def all_steps(self) -> Sequence[int]:
    return self._ckpt_mgr.all_steps()
