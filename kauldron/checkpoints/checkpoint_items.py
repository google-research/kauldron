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

"""Checkpoint handler."""

from __future__ import annotations

import abc
import functools
from typing import Any, ClassVar, Self

from etils import epy
from orbax import checkpoint as ocp

_State = Any


class CheckpointItem(abc.ABC):
  """Interface for a checkpoint item.

  This interface defines how instances should be saved by orbax.

  Protocol is equivalent to:

  ```python
  item = MyCheckpointItem(values=...)  # Subclass of `CheckpointItem`

  ckpt = ocp.CheckpointManager(item_handlers=item.__kd_ocp_handlers__())
  ckpt.save(step, args=item.__kd_ocp_save_args__())

  out = ckpt.restore(step, args=item.__kd_ocp_restore_args__())
  restored_item = item.__kd_ocp_restore_post__(out)
  ```
  """

  @abc.abstractmethod
  def __kd_ocp_handlers__(self) -> ocp.CheckpointHandler:
    """Handlers for `ocp.CheckpointManager(item_handlers=)`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    """Args passed to `ocp.CheckpointManager.save(args=)`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    """Args passed to `ocp.CheckpointManager.restore(args=)`."""
    raise NotImplementedError()

  def __kd_ocp_restore_post__(self, value: Any) -> Self:
    """Post-processing after `ocp.CheckpointManager.restore(args=)`."""
    return value


class TopLevelCheckpointItem(CheckpointItem):
  """Checkpoint item that contains other sub-checkpoint items.

  Usage:

  ```python
  class CheckpointState(typing.NamedTuple, TopLevelCheckpointItem):
    train_state: CheckpointItem
    metadata: CheckpointItem
    ds: CheckpointItem
  ```

  Usually it would be good practice to make this a `dataclass(kw_only=True)`.
  However in this case, it's convenient to be able to use it like a
  `NamedTuple`, like:

  ```python
  state, timer = ckptr.restore(CheckpointState(state, timer))
  ```

  Attributes:
    DEFAULT_ITEM: Default orbax item (restored if nothing is specified).
  """

  DEFAULT_ITEM: ClassVar[str | None] = None

  @functools.cached_property
  def _items_fields(self) -> dict[str, CheckpointItem]:
    fields = {name: getattr(self, name) for name in type(self)._fields}  # pytype: disable=attribute-error
    if self.DEFAULT_ITEM is not None:
      fields[ocp.checkpoint_manager.DEFAULT_ITEM_NAME] = fields.pop(
          self.DEFAULT_ITEM
      )
    return fields

  def __kd_ocp_handlers__(self) -> Any:  # -> ocp.CheckpointHandler
    # TODO(b/327866424): orbax implementation will bug if returning
    # `ocp.CompositeCheckpointHandler`, so have to return a dict instead.
    return {k: v.__kd_ocp_handlers__() for k, v in self._items_fields.items()}
    # return ocp.CompositeCheckpointHandler(**handlers)

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return ocp.args.Composite(
        **{k: v.__kd_ocp_save_args__() for k, v in self._items_fields.items()}
    )

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    return ocp.args.Composite(**{
        k: v.__kd_ocp_restore_args__() for k, v in self._items_fields.items()
    })

  def __kd_ocp_restore_post__(self, value: ocp.args.Composite) -> Self:
    # TODO(b/327882553): Orbax `ocp.args.Composite` look like a dict but behave
    # inconsistently (`__iter__`), so have to convert to dict first.
    value = dict(value.items())
    init_kwargs = {
        k: obj.__kd_ocp_restore_post__(val)
        for k, (obj, val) in epy.zip_dict(self._items_fields, value)
    }
    if self.DEFAULT_ITEM is not None:
      init_kwargs[self.DEFAULT_ITEM] = init_kwargs.pop(
          ocp.checkpoint_manager.DEFAULT_ITEM_NAME
      )
    return type(self)(**init_kwargs)


class StandardCheckpointItem(CheckpointItem):
  """Standard checkpoint item (for arbitrary `jax.Array` pytree).

  Inheriting from this class add support for checkpointing. Usage:

  ```python
  @flax.struct.dataclass
  class MyState(StandardCheckpointItem):
    params: Tree[jax.Array]
  ```

  Passing this base class to `Checkpointer.restore` allow to restore the state
  without knowing its structure.
  """

  def __kd_ocp_handlers__(self) -> ocp.CheckpointHandler:
    return ocp.StandardCheckpointHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return ocp.args.StandardSave(self)

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    # Allow to restore even when structure is not known
    if type(self) == StandardCheckpointItem:  # pylint: disable=unidiomatic-typecheck
      struct = None  # Not a subclass
    else:
      struct = self
    return ocp.args.StandardRestore(struct)
