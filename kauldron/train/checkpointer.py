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

# import abc
import dataclasses
import datetime
import functools
from typing import Optional, Sequence, TypeVar

from etils import epath
from flax.training import orbax_utils
import jax
from kauldron.utils import config_util
import orbax.checkpoint as ocp

_T = TypeVar("_T")


# TODO(epot): Why `abc` fail ?
class BaseCheckpointer(config_util.UpdateFromRootCfg):  # , abc.ABC):
  """Basic checkpointing interface."""

  # @abc.abstractmethod
  def restore(
      self,
      initial_state: _T,
      step: int = -1,
      *,
      noop_if_missing: bool = False,
  ) -> _T:
    raise NotImplementedError()

  # @abc.abstractmethod
  def should_save(self, step: int) -> bool:
    raise NotImplementedError()

  # @abc.abstractmethod
  def save_state(
      self,
      state,
      step: int,
      *,
      force: bool = False,
  ) -> bool:
    raise NotImplementedError()

  # @abc.abstractmethod
  def maybe_save_state(
      self,
      state,
      step: int,
      *,
      force: bool = False,
  ) -> bool:
    raise NotImplementedError()

  @property
  def latest_step(self) -> Optional[int]:
    return None

  @property
  def all_steps(self) -> Sequence[int]:
    return []


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class Checkpointer(BaseCheckpointer):
  """Basic Orbax Checkpointmanager."""
  workdir: str | epath.Path = config_util.ROOT_CFG_REF.workdir

  save_interval_steps: int
  max_to_keep: Optional[int] = 3
  keep_time_interval: Optional[datetime.timedelta] = None
  keep_period: Optional[int] = None

  fast: bool = True

  @functools.cached_property
  def _ckpt_mgr(self) -> ocp.checkpoint_manager.CheckpointManager:
    """Returns checkpoint manager instance (initialized and cached)."""
    mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=self.save_interval_steps,
        max_to_keep=self.max_to_keep,
        keep_time_interval=self.keep_time_interval,
        keep_period=self.keep_period,
        step_prefix="ckpt",
        # TODO(msajjadi): Re-enable this once we've figured it out.
        # step_format_fixed_length=9,
        create=True,
    )
    if self.fast:
      manager_cls = FastCheckpointManager
    else:
      manager_cls = ocp.CheckpointManager
    ckpt_mgr = manager_cls(
        epath.Path(self.workdir) / "checkpoints",
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
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


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class NoopCheckpointer(BaseCheckpointer):
  """Does nothing."""

  def restore(
      self, initial_state, step: int = -1, *, noop_if_missing: bool = False
  ):
    return initial_state

  def should_save(self, step: int) -> bool:
    return False

  def save_state(self, state, step: int, *, force: bool = False) -> bool:
    return False

  def maybe_save_state(self, state, step: int, *, force: bool = False) -> bool:
    if force:
      raise ValueError("NooCheckpointer cannot be forced to save.")
    return False


class FastCheckpointManager(ocp.CheckpointManager):
  """Wrapper around Checkpointmanager that speeds up loading."""

  def all_steps(self, read: bool = False) -> Sequence[int]:
    """Returns all steps tracked by the manager.

    Args:
      read: If True, forces a read directly from the storage location.
        Otherwise, a cached result can be returned.

    Returns:
      A sequence of steps (integers)
    """
    if read:
      return _checkpoint_steps(self.directory)
    return [ckpt.step for ckpt in self._checkpoints]


def _checkpoint_steps(checkpoint_dir: epath.PathLike) -> list[int]:
  """Returns a list of all steps for which a checkpoint exists in dir."""
  # Speeds up the original implementation by skipping the exists() and
  # is_directory() checks which trigger a CNS read for each checkpoint.
  checkpoint_dir = epath.Path(checkpoint_dir)

  def get_step_for_dir(step_dir: epath.Path) -> int:
    name = step_dir.name
    if ocp.utils.TMP_DIR_SUFFIX in name:
      return -1
    if name.isdigit():
      return int(name)
    _, _, suffix = name.rpartition("_")
    if suffix.isdigit():
      return int(suffix)
    return -1  # silently ignore directory/file

  steps = [get_step_for_dir(step_dir) for step_dir in checkpoint_dir.iterdir()]
  return [step for step in steps if step >= 0]
