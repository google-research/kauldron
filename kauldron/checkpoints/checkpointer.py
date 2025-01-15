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
import time
from typing import Any, Iterable, Optional, Self, Sequence, TypeVar

from absl import logging
from etils import epath
from etils import epy
import jax
from kauldron.checkpoints import checkpoint_items
from kauldron.checkpoints import lazy_checkpoint_manager
from kauldron.utils import config_util
import orbax.checkpoint as ocp

_State = checkpoint_items.CheckpointItem
_FnT = TypeVar("_FnT", bound=Callable[..., Any])
_StateT = TypeVar("_StateT", bound=_State)

CHECKPOINT_FOLDER_NAME = "checkpoints"

# pylint: disable=logging-fstring-interpolation


class BaseCheckpointer(
    config_util.UpdateFromRootCfg, epy.ContextManager, abc.ABC
):
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

  def reload(self) -> None:
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

  def close(self) -> None:
    """Closes the checkpointer."""
    pass

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()


def _retry(
    num_retries=3, retry_delay_seconds=5, exceptions=(Exception,)
) -> Callable[[_FnT], _FnT]:
  """Decorator for retrying a function call upon exceptions.

  Args:
      num_retries (int, optional): Maximum number of retries. Defaults to 3.
      retry_delay_seconds (int, optional): Delay in seconds between retries.
        Defaults to 5.
      exceptions (tuple, optional): Tuple of exception types to catch. Defaults
        to (Exception,).

  Returns:
    A decorator that retries the function call upon exceptions.
  """

  def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      for attempt in range(num_retries + 1):
        try:
          return func(*args, **kwargs)
        except exceptions as e:
          logging.exception(
              f"Exception occurred: {e}. Retrying (attempt"
              f" {attempt + 1}/{num_retries})..."
          )
          if attempt < num_retries:
            time.sleep(retry_delay_seconds)
          else:
            raise

    return wrapper

  return decorator


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class Checkpointer(BaseCheckpointer):
  """Wrapper around Orbax CheckpointManager.

  Attributes:
    workdir: Root directory of the task
    save_interval_steps: See `ocp.CheckpointManagerOptions`
    max_to_keep: See `ocp.CheckpointManagerOptions`
    keep_time_interval: See `ocp.CheckpointManagerOptions`
    keep_period: See `ocp.CheckpointManagerOptions`
    save_on_steps: See `ocp.CheckpointManagerOptions`
    multiprocessing_options: See `ocp.MultiprocessingOptions`
    fast: (internal) Activate some optimizations
    create: (internal) Whether to create the checkpoint directory, this is set
      by kauldron automatically based on whether the job is a training job
      (True) or an eval job (False).
  """

  workdir: epath.PathLike = config_util.ROOT_CFG_REF.workdir

  save_interval_steps: int
  max_to_keep: Optional[int] = 3
  keep_time_interval: Optional[datetime.timedelta] = None
  keep_period: Optional[int] = None
  save_on_steps: Optional[Sequence[int]] = None
  multiprocessing_options: ocp.options.MultiprocessingOptions = (
      dataclasses.field(default_factory=ocp.options.MultiprocessingOptions)
  )

  fast: bool = True
  create: bool = True

  @functools.cached_property
  def _ckpt_mgr(self) -> lazy_checkpoint_manager.LazyCheckpointManager:
    """Returns checkpoint manager instance (initialized and cached)."""
    mgr_options = ocp.CheckpointManagerOptions(
        save_interval_steps=self.save_interval_steps,
        max_to_keep=self.max_to_keep,
        keep_time_interval=self.keep_time_interval,
        keep_period=self.keep_period,
        save_on_steps=self.save_on_steps,
        step_prefix="ckpt",
        # TODO(msajjadi): Re-enable this once we've figured it out.
        # step_format_fixed_length=9,
        create=self.create,
        # TODO(epot): Add `best_fn` to allow `ckpt_mngr.best_step()`
        async_options=ocp.AsyncOptions(
            timeout_secs=60 * 30,  # 30 minutes
        ),
        multiprocessing_options=self.multiprocessing_options,
        # Ensure that checkpoints are not world-readable.
        # This file mode removes permission bits for OTHER in the POSIX format.
        # See
        #/checkpoint#do-not-set-the-checkpoint-directory-to-be-world-readable
        file_options=ocp.checkpoint_manager.FileOptions(
            path_permission_mode=0o770,
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
        jax.tree.map(_release_memory, state)

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

  def delete(self, step: int) -> None:
    return self._ckpt_mgr.delete(step)

  # TODO(b/299684331) workaround for the flaky orbax checkpointing.
  @_retry(num_retries=3, exceptions=(ValueError,))
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

  # TODO(b/330748987): Remove the loop. Currently required because `.reload`
  # sometimes crashes during race conditions.
  @_retry(num_retries=5, exceptions=(OSError,))
  def reload(self) -> None:
    self._ckpt_mgr.reload()

  def _absolute_step(self, step: int) -> int:
    """Convert `-1` into the last step."""
    step = self._ckpt_mgr.latest_step() if step == -1 else step
    # Do not check step as it can lead to race conditions.
    # if step not in self._ckpt_mgr.all_steps():
    #   raise ValueError(f"No checkpoint is available for step {step}")
    return step  # pytype: disable=bad-return-type

  def item_metadata(self, step: int = -1) -> dict[str, Any]:
    """Returns the metadata (tree, shape,...) associated with the step."""
    step = self._absolute_step(step)
    # Warning: `item_metadata` can be `None` if the step does not exists.
    return self._ckpt_mgr.item_metadata(step)

  def iter_new_checkpoints(
      self,
      *,
      min_interval_secs: int = 0,
      timeout: Optional[int] = None,
      timeout_fn: Optional[Callable[[], bool]] = None,
  ) -> Iterator[int]:
    num_retries = 4
    attempt = num_retries
    while True:
      try:
        for step in self._ckpt_mgr.iter_new_checkpoints(
            min_interval_secs=min_interval_secs,
            timeout=timeout,
            timeout_fn=timeout_fn,
        ):
          self.reload()
          yield step
          attempt = num_retries  # After every successful steps, reset.
          # Refresh the checkpoint manager cache used by `.all_steps()`
          # (b/315316885#6)
        return  # If finish, exit
      except ValueError:
        if attempt > 0:
          logging.exception(
              f"Retrying (remaining attempt {attempt})...Retrying..."
          )
          attempt -= 1
          time.sleep(3)
        else:
          raise

  def wait_until_finished(self) -> None:
    self._ckpt_mgr.wait_until_finished()

  def close(self) -> None:
    self._ckpt_mgr.close()


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
