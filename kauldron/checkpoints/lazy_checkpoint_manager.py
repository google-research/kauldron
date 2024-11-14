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

"""Lazy checkpoint manager."""

from collections.abc import Callable, Iterator
import dataclasses
from typing import Any, Optional, TypeVar

from etils import epath
from kauldron.checkpoints import checkpoint_items
from orbax import checkpoint as ocp

_State = checkpoint_items.CheckpointItem
_StateT = TypeVar("_StateT", bound=_State)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LazyCheckpointManager:
  """Wrapper around `ocp.CheckpointManager` that defines the handler lazyly.

  Orbax API attempts to be lazy with `ocp.args`, but fail with various bugs (
  b/320668278, b/320674317, `ocp.CompositeCheckpointHandler` names required
  at initialization time). Summarized in b/328149581.

  This class make the checkpointer truely lazy. This is done by creating
  the `ocp.CheckpointManager` only during the first `.save` / `.restore` call.

  API is also simplified. Save/restore item should be a
  `kd.ckpt.items.CheckpointItem` that implement the kauldron orbax protocol.
  No need to wrap items inside `ocp.args.StandardArgs`,...
  """

  directory: epath.PathLike
  _: dataclasses.KW_ONLY
  options: ocp.CheckpointManagerOptions
  fast: bool = True

  # Those arguments are dynamically set on first usage (in `_create_manager`)
  # Note that it's possible that when `.latest_step()`,... are called the
  # `_item_handlers` might know yet be known. Then a `_mgr` might be created
  # without handler information, and `_mgr` get replaced once the informations
  # are known.
  _mgr: ocp.CheckpointManager | None = None
  _item_handlers: ocp.CheckpointHandler | None = None

  def _get_manager(self, state: _State | None = None) -> ocp.CheckpointManager:
    """Returns the `_mgr` (and create it if needed)."""
    # Create the CheckpointManager if:
    # * There's not checkpoint manager yet (first call)
    # * There's a checkpoint manager but only empty (don't contain handler
    #   information)
    if self._mgr is None or (self._item_handlers is None and state is not None):
      self._create_manager(state)

    # TODO(epot): Should validate the state and self._mgr item_handlers match.

    assert self._mgr is not None
    return self._mgr

  def _create_manager(self, state: _State | None) -> None:
    """Assign the `_mgr` and `_item_handlers` variables."""
    # Maybe set `_item_handlers`
    if state is not None:
      assert self._item_handlers is None
      item_handlers = state.__kd_ocp_handlers__()
      object.__setattr__(self, "_item_handlers", item_handlers)

    # Create root directory.
    # This avoids triggering a race-condition when multiple orbax
    # CheckpointManager jobs are trying to create the same checkpoint directory
    # at the same time.
    if self.options.create:
      epath.Path(self.directory).mkdir(parents=True, exist_ok=True)

    # Set `_mgr`
    if self.fast:
      manager_cls = _FastCheckpointManager
    else:
      manager_cls = ocp.CheckpointManager
    mgr = manager_cls(
        directory=self.directory,
        options=self.options,
        item_handlers=self._item_handlers,
    )
    object.__setattr__(self, "_mgr", mgr)

  def save(
      self,
      state: _State,
      *,
      step: int,
      force: bool = False,
  ) -> bool:
    """Wrapper around `ocp.CheckpointManager.save`."""
    mgr = self._get_manager(state)
    mgr.save(
        step,
        args=state.__kd_ocp_save_args__(),
        force=force,
    )  # pytype: disable=bad-return-type

  def restore(self, state: _StateT, *, step: int) -> _StateT:
    """Wrapper around `ocp.CheckpointManager.restore`."""
    mgr = self._get_manager(state)
    restored = mgr.restore(step=step, args=state.__kd_ocp_restore_args__())
    return state.__kd_ocp_restore_post__(restored)

  def item_metadata(self, step: int = -1) -> dict[str, Any]:
    """Wrapper around `ocp.CheckpointManager.item_metadata`."""
    if self._item_handlers is None:
      # TODO(b/320668278) orbax does not support restoring the metadata
      # when the handler isn't known. Currently we hardcode standard checkpoint
      # item as the default handler but we could support generic handlers,
      # by saving the item_handler mapping (name <> type).
      state = checkpoint_items.StandardCheckpointItem()
    else:
      state = None
    # TODO(epot): Could eventually support restoring metadata for all handlers.
    mgr = self._get_manager(state)

    return mgr.item_metadata(step)

  def should_save(self, step) -> bool:
    """Wrapper around `ocp.CheckpointManager.should_save`."""
    return self._get_manager().should_save(step)

  def delete(self, step: int) -> None:
    """Wrapper around `ocp.CheckpointManager.delete`.

    Args:
      step: The step to delete.
    """
    self._get_manager().delete(step)

  def latest_step(self) -> int | None:
    """Wrapper around `ocp.CheckpointManager.latest_step`."""
    return self._get_manager().latest_step()

  def all_steps(self) -> list[int]:
    """Wrapper around `ocp.CheckpointManager.all_steps`."""
    return list(self._get_manager().all_steps())

  def reload(self) -> None:
    """Wrapper around `ocp.CheckpointManager.reload`."""
    return self._get_manager().reload()

  def wait_until_finished(self) -> None:
    """Wrapper around `ocp.CheckpointManager.wait_until_finished`."""
    return self._get_manager().wait_until_finished()

  def iter_new_checkpoints(
      self,
      *,
      min_interval_secs: int = 0,
      timeout: Optional[int] = None,
      timeout_fn: Optional[Callable[[], bool]] = None,
  ) -> Iterator[int]:
    """Wrapper around `ocp.checkpoint_utils.checkpoints_iterator`."""
    mgr = self._get_manager()

    # TODO(b/315316885): This should be part of the `ocp.CheckpointManager`
    # API (see comment 4)
    for step in ocp.checkpoint_utils.checkpoints_iterator(
        checkpoint_dir=mgr.directory,
        step_prefix=mgr._options.step_prefix,  # pylint: disable=protected-access
        step_format_fixed_length=mgr._options.step_format_fixed_length,  # pylint: disable=protected-access
        min_interval_secs=min_interval_secs,
        timeout=timeout,
        timeout_fn=timeout_fn,
    ):
      yield step

  def close(self) -> None:
    """Wrapper around `ocp.CheckpointManager.close`."""
    return self._get_manager().close()


class _FastCheckpointManager(ocp.CheckpointManager):
  """Wrapper around Checkpointmanager that speeds up loading."""

  def all_steps(self, read: bool = False) -> list[int]:
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
