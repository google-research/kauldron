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
import dataclasses
import functools
from typing import Any, TypeVar

from etils import epath
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import jax
from kauldron import kontext
from kauldron.checkpoints import checkpoint_items
from kauldron.checkpoints import checkpointer
from kauldron.utils import xmanager
import orbax.checkpoint as ocp

with epy.lazy_imports():
  from orbax.checkpoint.experimental.model_surgery import standard_checkpoint_handler  # pylint: disable=g-import-not-at-top

_T = TypeVar('_T')


class _NOT_RESTORED:  # pylint: disable=invalid-name
  """Sentinel to check `.restore()` checkpoint structure output."""

  pass


_NOT_RESTORED = _NOT_RESTORED()


# TODO(klausg): maybe rename, since this is now only used in cfg.init_transforms
# TODO(klausg): move out of checkpoints?
class AbstractPartialLoader(abc.ABC):
  """Abstract class for partial checkpoint loaders."""

  @abc.abstractmethod
  def transform(self, state: _T) -> _T:
    """Transform the state by updating it with pre-trained values.

    Note: `transform` functions can modify the `state` values but should NOT
    modify its structure, shape or dtypes.

    Args:
      state: The `state` object to transform

    Returns:
      The updated `state`
    """
    raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class PartialKauldronLoader(AbstractPartialLoader):
  """Parial loader for Kauldron checkpoints.

  Allow to use pretrained weights from another Kauldron checkpoint.

  Usage:

  ```python
  cfg.init_transforms = {
      'pretrained_init': kd.ckpts.PartialKauldronLoader(
          workdir='/path/to/original/work_unit/',
          new_to_old={  # Mapping params
              # '<new_path>':            '<source_path>'
              'params.decoder.layers_0': 'params.endoder',
          },
      )
  }

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
      checkpoint.
    step: Which step to load (default to last one)
  """

  workdir: epath.PathLike
  new_to_old: dict[str, str]  # PyTree[types.EllipsisType] = ...
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


@dataclasses.dataclass(frozen=True)
class _PartialRestoreCheckpointItem(checkpoint_items.CheckpointItem):
  """Restore a partial checkpoint."""

  state: Any
  _: dataclasses.KW_ONLY
  new_to_old: dict[str, str]  # PyTree[types.EllipsisType] = ...

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


def workdir_from_xid(xid: int, wid: int = 1) -> epath.Path:
  return xmanager.Experiment.from_xid(xid=xid, wid=wid).wu.workdir


@dataclasses.dataclass(frozen=True, kw_only=True)
class PartialLoader(AbstractPartialLoader):
  """Specify where to load the weights to initialize a model.

  Allow to use pretrained weights from another model.

  Usage:

  ```python
  cfg.init_transforms = {
      'pretrained_init': kd.ckpts.PartialLoader(
          source=kd.ckpts.KauldronSource('/path/to/original/work_unit/'),
          new_to_old={  # Mapping params
              # '<new_path>':            '<source_path>'
              'params.decoder.layers_0': 'params.endoder',
          },
      )
  }

  trainer = konfig.resolve(cfg)

  # When initializing the weights, the `init_transform` is applied
  init_state = trainer.init_state()

  # `init_state.params['decoder']['layers_0']` now contain the previous encoder
  # weights
  ```

  Attributes:
    source: From where to load the checkpoint (kauldron, xid, external
      project,...)
    new_to_old: Mapping the pytree to copy to the new state from the original
      checkpoint.
  """

  source: CkptSource
  new_to_old: dict[str, str]  # PyTree[types.EllipsisType] = ...

  def __post_init__(self):
    # Normalize new_to_old
    # This is required because `ConfigDict({'a.b': ...})` fail
    # Use `type()` to propagate the `ImmutableDict`
    new_to_old = type(self.new_to_old)({
        k.replace('/', '.'): v.replace('/', '.')
        for k, v in self.new_to_old.items()
    })
    object.__setattr__(self, 'new_to_old', new_to_old)

  @functools.cached_property
  def _new_to_old_path(self) -> dict[kontext.Path, kontext.Path]:
    return {
        kontext.Path.from_str(from_): kontext.Path.from_str(to)
        for from_, to in self.new_to_old.items()
    }

  def transform(self, state: _T) -> _T:
    """Transform the state by updating it with pre-trained values."""
    # Deep copy (mutable)
    # `orbax` uses nested `dict`, rather than PyTree
    to_state = ocp.utils.serialize_tree(state, keep_empty_nodes=True)

    # TODO(epot): Should refactor once Orbax support the new transform API

    from_state = self.source.metadata()

    # For each state to copy, make sure the to/from matches
    for to_path, from_path in self._new_to_old_path.items():
      to_state_inner = to_path.get_from(to_state, err_spec=True)
      from_state_inner = from_path.get_from(from_state, err_spec=True)

      # TODO(epot): Restore once flax `save_args_from_target` are removed.
      to_specs = jax.tree.map(lambda x: ..., to_state_inner)
      from_specs = jax.tree.map(lambda x: ..., from_state_inner)
      # to_specs = etree.spec_like(to_state_inner)
      # from_specs = etree.spec_like(from_state_inner)

      if from_specs != to_specs:
        raise ValueError(
            f'Imported module structure do not match for {from_path} to'
            f' {to_path}:\n'
            f'From:  {epy.pretty_repr(from_specs)}\n'
            f'Target: {epy.pretty_repr(to_specs)}\n'
            f'Diff: {epy.diff_str(from_specs, to_specs)}'
        )

    # Compute the tree struct to extract:
    # * Avoid restoring the full tree (only the requested weights)
    # * Allow computing the `RestoreArgs` (matching the target state)
    # Maybe there's a way of implementing this using `ocp.apply_transformations`
    # but I'm not smart enough. It looks like passing `transforms={}` to
    # `ckpt.restore` fail to perform the proper renaming.
    from_state_subtree = _extract_sub_tree(to_state, self._new_to_old_path)

    # `from` and `to` state
    from_state = self.source.restore(from_state_subtree)

    # For each state to copy
    for to_path, from_path in self._new_to_old_path.items():
      from_state_inner = from_path.get_from(from_state)
      to_state_inner = to_path[:-1].get_from(to_state)

      # Replace the state
      to_state_inner[to_path[-1]] = from_state_inner

    return ocp.utils.deserialize_tree(
        to_state, target=state, keep_empty_nodes=True
    )


class CkptSource(abc.ABC):
  """Partial checkpoint loader source. See `kd.ckpts.PartialLoader`."""

  @abc.abstractmethod
  def metadata(self) -> Any:
    """Returns the `orbax` metadata (checkpoint structure)."""
    raise NotImplementedError

  @abc.abstractmethod
  def restore(self, item) -> Any:
    """Restore the state.

    Args:
      item: Specify the items to restore. Other items will be ignored.

    Returns:
      The extracted items.
    """
    raise NotImplementedError


# TODO(epot): Re-write using the ocp transform API.
@dataclasses.dataclass(frozen=True)
class KauldronSource(CkptSource):
  """Kauldron loader source for `kd.ckpts.PartialLoader`.

  Attributes:
    workdir: The work directory from which the checkpoint should be loaded
    step: Which step to load (default to last one)
  """

  workdir: epath.PathLike

  _: dataclasses.KW_ONLY

  step: int = -1

  def __post_init__(self):
    object.__setattr__(self, 'workdir', epath.Path(self.workdir))

  @classmethod
  def from_xid(
      cls, xid: int, wid: int = 1, *, step: int = -1
  ) -> KauldronSource:
    """Creates a KauldronSource from an xid."""
    return cls(
        workdir=workdir_from_xid(xid=xid, wid=wid),
        step=step,
    )

  @functools.cached_property
  def _ckpt_mgr(self) -> checkpointer.Checkpointer:
    return checkpointer.Checkpointer(
        workdir=self.workdir,
        save_interval_steps=1,
    )

  def metadata(self) -> Any:
    metadata = self._ckpt_mgr.item_metadata(self.step)
    if metadata is None:
      raise ValueError(f'No metadata found for step: {self.step}')
    return metadata

  def restore(self, item) -> Any:
    """Loads the params from the checkpoint."""
    # TODO(epot): Should only restore the required params, not everything
    state = self._ckpt_mgr.restore(
        checkpoint_items.StandardCheckpointItem(), step=self.step
    )

    # Validate `state` do not contain `_NOT_RESTORED`
    _assert_all_restored(state)

    return state


def _assert_all_restored(state: Any) -> None:
  has_non_restored = False

  def _check_is_restored(v):
    nonlocal has_non_restored
    if v is _NOT_RESTORED:
      has_non_restored = True

  jax.tree.map(_check_is_restored, state)
  if has_non_restored:
    raise ValueError(
        'Restored structure do not match expected:'
        f' {epy.pretty_repr(etree.spec_like(state))}'
    )


def _extract_sub_tree(to_state, new_to_old_path):
  """Extract the subtree of value."""
  # Limitation:
  # * This do not support `list`

  struct_to_extract = {}
  for to_path, from_path in new_to_old_path.items():
    to_values = to_path.get_from(to_state, err_spec=True)

    curr_struct = struct_to_extract
    for part in from_path[:-1]:
      if part not in curr_struct:
        if not isinstance(part, str):
          raise NotImplementedError(
              f'Structures with list not supported: {from_path}'
          )
        curr_struct[part] = {}
      curr_struct = curr_struct[part]
    if from_path[-1] in curr_struct:
      raise ValueError(f'Overwriting {from_path} (already set previously).')
    curr_struct[from_path[-1]] = to_values
  return struct_to_extract
