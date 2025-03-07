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

"""Partial checkpoint utils."""

from __future__ import annotations

import abc
from collections.abc import MutableMapping
import dataclasses
import functools
import typing
from typing import Any, Iterable, Self, TypeVar

from etils import epath
from etils import epy
import flax
from kauldron import kontext
from kauldron.checkpoints import checkpoint_items
from kauldron.checkpoints import checkpointer
from kauldron.utils import from_xid
import orbax.checkpoint as ocp

with epy.lazy_imports():
  from orbax.checkpoint.experimental.model_surgery import standard_checkpoint_handler  # pylint: disable=g-import-not-at-top

FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict
_T = TypeVar('_T')


# TODO(epot): rename to `InitTransform`
class AbstractPartialLoader(abc.ABC):
  """Abstract class for partial checkpoint loaders."""

  @abc.abstractmethod
  def transform(self, state: _T) -> _T:
    """Transform the state by updating it with pre-trained values.

    Notes:

    * `transform` functions can modify the `state` values but should NOT
      modify its structure, shape or dtypes.
    * `transform` should correctly propagate the sharding information from the
      given state.

    Args:
      state: The `state` object to transform

    Returns:
      The updated `state`
    """
    raise NotImplementedError


class NoopTransform(AbstractPartialLoader):
  """`init_transform` that does nothing."""

  def transform(self, state):
    return state


class MultiTransform(AbstractPartialLoader):
  """Transform which applies multiple transformations sequentially."""

  def __init__(self, **transforms: AbstractPartialLoader):
    self._transforms = transforms

  def transform(self, state):
    for k, tr in self._transforms.items():
      try:
        state = tr.transform(state)
      except Exception as e:  # pylint: disable=broad-exception-caught
        epy.reraise(e, prefix=f'init_transform {k!r} failed: ')
    return state


@dataclasses.dataclass(frozen=True, kw_only=True)
class PartialKauldronLoader(epy.ContextManager, AbstractPartialLoader):
  """Partial loader for Kauldron checkpoints.

  Allow to use pretrained weights from another Kauldron checkpoint.

  Usage:

  ```python
  cfg.init_transform = kd.ckpts.PartialKauldronLoader(
      workdir='/path/to/original/work_unit/',
      new_to_old={  # Mapping params
          # '<new_path>':            '<source_path>'
          'params.decoder.layers_0': 'params.encoder',
      },
  )

  trainer = konfig.resolve(cfg)

  # When initializing the weights, the `init_transform` is applied
  init_state = trainer.init_state()

  # `init_state.params['decoder']['layers_0']` now contain the previous encoder
  # weights
  ```

  Attributes:
    workdir: The work directory from which the checkpoint should be loaded ( can
      be created from `kd.ckpts.workdir_from_xid`).
    new_to_old: Mapping the pytree to copy to the new state from the original
      checkpoint. By default, copy all model `params` and `collections`
    step: Which step to load (default to last one)
  """

  workdir: epath.PathLike
  new_to_old: MutableMapping[str, str] = dataclasses.field(
      default_factory=lambda: FrozenDict({
          'params': 'params',
          'collections': 'collections',
      })
  )
  step: int = -1

  def transform(self, state: _T) -> _T:
    # This could potentially be extended to non-Kauldron checkpoints (like
    # `kd.ckpts.PartialOrbaxLoader`)
    return self._ckpt_mgr.restore(
        _PartialRestoreCheckpointItem(state, new_to_old=self.new_to_old),
        step=self.step,
    )

  @functools.cached_property
  def _ckpt_mgr(self) -> checkpointer.Checkpointer:
    return checkpointer.Checkpointer(
        workdir=self.workdir,
        save_interval_steps=1,
    )

  def close(self) -> None:
    self._ckpt_mgr.close()

  def __contextmanager__(
      self,
  ) -> Iterable[Self]:
    try:
      yield self
    finally:
      self.close()


@dataclasses.dataclass(frozen=True)
class _PartialRestoreCheckpointItem(checkpoint_items.CheckpointItem):
  """Restore a partial checkpoint."""

  state: Any
  _: dataclasses.KW_ONLY
  new_to_old: MutableMapping[str, str]

  def __kd_ocp_handlers__(self) -> ocp.CheckpointHandler:
    return standard_checkpoint_handler.StandardCheckpointHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    raise ValueError('Partial checkpoint do not support save.')

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    # Extract the sub-tree from the new state
    sub_state = {
        new_path: kontext.get_by_path(self.state, new_path)
        for new_path, _ in self.new_to_old.items()
    }
    return standard_checkpoint_handler.StandardRestoreArgs(
        sub_state, transform_fn=self._transform_fn
    )

  def __kd_ocp_restore_post__(self, partial_restore: Any):
    """Post-processing after `ocp.CheckpointManager.restore(args=)`."""
    # Mutate the state to overwrite the sub-tree with the partially restored
    # checkpoint.

    # Deep copy (mutable)
    # `orbax` uses nested `dict`, rather than PyTree
    state = ocp.utils.serialize_tree(self.state, keep_empty_nodes=True)

    for new_path, _ in self.new_to_old.items():
      kontext.set_by_path(state, new_path, partial_restore[new_path])

    state = ocp.utils.deserialize_tree(
        state, target=self.state, keep_empty_nodes=True
    )
    return state

  def _transform_fn(self, restored):
    # Extract the sub-tree from the old state
    sub_state = {
        new_path: kontext.get_by_path(restored, old_path)
        for new_path, old_path in self.new_to_old.items()
    }
    return sub_state


# TODO(epot): Deprecate the alias and migrate everyone to
# `kd.from_xid.get_workdir`
def workdir_from_xid(xid: int, wid: int = 1) -> epath.Path:
  return from_xid.get_workdir(xid=xid, wid=wid)
