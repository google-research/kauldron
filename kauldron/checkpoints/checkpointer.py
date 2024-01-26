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

"""Implementations of Orbax checkpointers for typical usecases."""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator
import dataclasses
import datetime
import functools
from typing import Any, Optional, Sequence, TypeVar

from etils import epath
import jax
from kauldron.utils import config_util
import orbax.checkpoint as ocp

_T = TypeVar("_T")

CHECKPOINT_FOLDER_NAME = "checkpoints"


class BaseCheckpointer(config_util.UpdateFromRootCfg, abc.ABC):
  """Basic checkpointing interface.

  2 implementations:

  * `Checkpointer`: Wrapper around Orbax CheckpointManager.
  * `NoopCheckpointer`: Does nothing.
  """

  @abc.abstractmethod
  def restore(
      self,
      initial_state: _T | None = None,
      *,
      step: int = -1,
      noop_if_missing: bool = False,
  ) -> _T:
    raise NotImplementedError()

  @abc.abstractmethod
  def should_save(self, step: int) -> bool:
    raise NotImplementedError()

  @abc.abstractmethod
  def save(
      self,
      state,
      *,
      step: int,
      force: bool = False,
  ) -> bool:
    raise NotImplementedError()

  @abc.abstractmethod
  def maybe_save(
      self,
      state,
      *,
      step: int,
      force: bool = False,
  ) -> bool:
    raise NotImplementedError()

  @property
  def latest_step(self) -> Optional[int]:
    return None

  @property
  def all_steps(self) -> Sequence[int]:
    return []

  def refresh_cache(self) -> None:
    """Refresh the cache.

    For performance, the checkpointer caches the directory names. Calling this
    function resets the cache to allow scanning the checkpoint directory for new
    checkpoints.
    """
    pass

  def wait_until_finished(self) -> None:
    """Synchronizes the asynchronous checkpointing."""
    pass

  def iter_new_checkpoints(
      self,
      *,
      min_interval_secs: int = 0,
      timeout: Optional[int] = None,
      timeout_fn: Optional[Callable[[], bool]] = None,
  ) -> Iterator[int]:
    """Wrapper around `ocp.checkpoint_utils.checkpoints_iterator`."""
    del min_interval_secs, timeout, timeout_fn
    yield from []


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class Checkpointer(BaseCheckpointer):
  """Wrapper around Orbax CheckpointManager.

  Attributes:
    workdir: Root directory of the task
    save_interval_steps: See `ocp.CheckpointManagerOptions`
    max_to_keep: See `ocp.CheckpointManagerOptions`
    keep_time_interval: See `ocp.CheckpointManagerOptions`
    keep_period: See `ocp.CheckpointManagerOptions`
    fast: (internal) Activate some optimizations
  """

  workdir: str | epath.Path = config_util.ROOT_CFG_REF.workdir

  save_interval_steps: int
  max_to_keep: Optional[int] = 3
  keep_time_interval: Optional[datetime.timedelta] = None
  keep_period: Optional[int] = None

  fast: bool = True

  @functools.cached_property
  def _ckpt_mgr(self) -> ocp.CheckpointManager:
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
        # TODO(epot): Add `best_fn` to allow `ckpt_mngr.best_step()`
        # async_options=ocp.AsyncOptions(
        #     timeout_secs=60 * 30,  # 30 minutes
        # ),
    )
    if self.fast:
      manager_cls = FastCheckpointManager
    else:
      manager_cls = ocp.CheckpointManager
    ckpt_mgr = manager_cls(
        epath.Path(self.workdir) / CHECKPOINT_FOLDER_NAME,
        options=mgr_options,
    )
    return ckpt_mgr

  def restore(
      self,
      initial_state: _T | None = None,
      *,
      step: int = -1,
      noop_if_missing: bool = False,
  ) -> _T:
    """Restore state."""

    state = initial_state
    if self._ckpt_mgr.latest_step() is not None:
      step = self._absolute_step(step)

      if initial_state is None:
        args = ocp.args.StandardRestore()
      else:
        args = ocp.args.StandardRestore(initial_state)

      state = self._ckpt_mgr.restore(step, args=args)
    elif not noop_if_missing:  # No checkpoint
      raise FileNotFoundError(
          f"No checkpoint found in {self.workdir}. Use `noop_if_missing=True`"
          " to default to initial state."
      )
    return state

  def should_save(self, step: int) -> bool:
    return self._ckpt_mgr.should_save(step)

  def save(
      self,
      state,
      *,
      step: int,
      force: bool = False,
  ) -> bool:
    """Save state."""
    with jax.transfer_guard("allow"):
      return self._ckpt_mgr.save(
          step,
          args=ocp.args.StandardSave(state),
          force=force,
      )

  def maybe_save(
      self,
      state,
      *,
      step: int,
      force: bool = False,
  ) -> bool:
    """Save state."""
    if not self.should_save(step):
      return False

    return self.save(state, step=step, force=force)

  @property
  def latest_step(self) -> Optional[int]:
    return self._ckpt_mgr.latest_step()

  @property
  def all_steps(self) -> Sequence[int]:
    return self._ckpt_mgr.all_steps()

  def refresh_cache(self) -> None:
    # If cache is refreshed, we reset the checkpoint manager by replaying
    # the cached property.
    # TODO(b/315316885): Orbax should expose a native method for this
    object.__setattr__(self, "_ckpt_mgr", type(self)._ckpt_mgr.func(self))

  def _absolute_step(self, step: int) -> int:
    """Convert `-1` into the last step."""
    step = self._ckpt_mgr.latest_step() if step == -1 else step
    if step not in self._ckpt_mgr.all_steps():
      raise ValueError(f"No checkpoint is available for step {step}")
    return step  # pytype: disable=bad-return-type

  def item_metadata(self, step: int = -1) -> dict[str, Any]:
    """Returns the metadata (tree, shape,...) associated with the step."""
    step = self._absolute_step(step)
    return self._ckpt_mgr.item_metadata(step)

  def iter_new_checkpoints(
      self,
      *,
      min_interval_secs: int = 0,
      timeout: Optional[int] = None,
      timeout_fn: Optional[Callable[[], bool]] = None,
  ) -> Iterator[int]:
    for step in ocp.checkpoint_utils.checkpoints_iterator(
        checkpoint_dir=self._ckpt_mgr.directory,
        step_prefix=self._ckpt_mgr._options.step_prefix,  # pylint: disable=protected-access
        step_format_fixed_length=self._ckpt_mgr._options.step_format_fixed_length,  # pylint: disable=protected-access
        min_interval_secs=min_interval_secs,
        timeout=timeout,
        timeout_fn=timeout_fn,
    ):
      yield step

  def wait_until_finished(self) -> None:
    self._ckpt_mgr.wait_until_finished()


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class NoopCheckpointer(BaseCheckpointer):
  """Does nothing."""

  def restore(
      self, initial_state=None, *, step: int = -1, noop_if_missing: bool = False
  ):
    if initial_state is None:
      raise ValueError("`NooCheckpointer.restore` require the state arg.")
    return initial_state

  def should_save(self, step: int) -> bool:
    return False

  def save(self, state, *, step: int, force: bool = False) -> bool:
    return False

  def maybe_save(self, state, *, step: int, force: bool = False) -> bool:
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
