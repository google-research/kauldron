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
from kauldron.checkpoints import checkpoint_items
from kauldron.checkpoints import lazy_checkpoint_manager
from kauldron.utils import config_util
import orbax.checkpoint as ocp

_State = checkpoint_items.CheckpointItem
_StateT = TypeVar("_StateT", bound=_State)

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
      state: _StateT,
      *,
      step: int = -1,
      noop_if_missing: bool = False,
      donate: bool = True,
  ) -> _StateT:
    raise NotImplementedError()

  @abc.abstractmethod
  def should_save(self, step: int) -> bool:
    raise NotImplementedError()

  @abc.abstractmethod
  def save(
      self,
      state: _State,
      *,
      step: int,
      force: bool = False,
  ) -> bool:
    raise NotImplementedError()

  @abc.abstractmethod
  def maybe_save(
      self,
      state: _State,
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
  def _ckpt_mgr(self) -> lazy_checkpoint_manager.LazyCheckpointManager:
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
        async_options=ocp.AsyncOptions(
            timeout_secs=60 * 30,  # 30 minutes
        ),
    )

    return lazy_checkpoint_manager.LazyCheckpointManager(
        directory=epath.Path(self.workdir) / CHECKPOINT_FOLDER_NAME,
        options=mgr_options,
        fast=self.fast,
    )

  def restore(
      self,
      state: _StateT,
      *,
      step: int = -1,
      noop_if_missing: bool = False,
      donate: bool = True,
  ) -> _StateT:
    """Restore state.

    Args:
      state: The `state` object initialized from the trainer. If the state is
        not known, you can pass `kd.ckpt.items.StandardCheckpointItem()` to
        restore the nested `dict` of weights.
      step: The training step of the checkpoint to restore. -1 means last step.
      noop_if_missing: If False will raise an error when no checkpoint is found.
      donate: Whether delete the `initial_state` to free up memory when
        restoring the checkpoint. This avoids 2x memory consumption. It is safe
        to donate the `initial_state` if you no longer need it after restoring.

    Returns:
      The restored `state`.

    Raises:
      FileNotFoundError: An error occurred when no checkpoint is found.
    """
    if self._ckpt_mgr.latest_step() is not None:
      step = self._absolute_step(step)

      # Delete `state` to free up memory.
      if donate:
        jax.tree_map(_release_memory, state)

      state = self._ckpt_mgr.restore(state, step=step)
    elif not noop_if_missing:  # No checkpoint
      raise FileNotFoundError(
          f"No checkpoint found in {self.workdir}. Use `noop_if_missing=True`"
          " to default to initial state."
      )
    # Otherwise returns the unchanged state (noop_if_missing == True)

    return state

  def should_save(self, step: int) -> bool:
    return self._ckpt_mgr.should_save(step)

  def save(
      self,
      state: _State,
      *,
      step: int,
      force: bool = False,
  ) -> bool:
    """Save state."""
    with jax.transfer_guard("allow"):
      return self._ckpt_mgr.save(
          state,
          step=step,
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
    self._ckpt_mgr.reload()

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
    for step in self._ckpt_mgr.iter_new_checkpoints(
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
      self,
      state,
      *,
      step: int = -1,
      noop_if_missing: bool = False,
      donate: bool = True,
  ):
    if state is None:
      raise ValueError("`NooCheckpointer.restore` require the state arg.")
    return state

  def should_save(self, step: int) -> bool:
    return False

  def save(self, state, *, step: int, force: bool = False) -> bool:
    return False

  def maybe_save(self, state, *, step: int, force: bool = False) -> bool:
    if force:
      raise ValueError("NooCheckpointer cannot be forced to save.")
    return False


def _release_memory(x):
  """Deletes and releases the memory of a Jax array."""
  if isinstance(x, jax.Array):
    x.delete()
  return x
