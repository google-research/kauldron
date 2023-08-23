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

"""Partial checkpoint utils."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Any, TypeVar

from etils import epath
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
from flax.training import orbax_utils
import jax
from kauldron.checkpoints import checkpointer
from kauldron.utils import paths as path_utils
import orbax.checkpoint as ocp

_T = TypeVar('_T')


class _NOT_RESTORED:  # pylint: disable=invalid-name
  """Sentinel to check `.restore()` checkpoint structure output."""

  pass


_NOT_RESTORED = _NOT_RESTORED()


class AbstractPartialLoader(abc.ABC):
  """Abstract class for partial checkpoint loaders."""

  @abc.abstractmethod
  def transform(self, state: _T) -> _T:
    """Transform the state by updating it with pre-trained values."""
    raise NotImplementedError


@dataclasses.dataclass(frozen=True, kw_only=True)
class PartialLoader(AbstractPartialLoader):
  """Specify where to load the weights to initialize a model.

  Allow to use pretrained weights from another model.

  Usage:

  ```python
  checkpoint = kd.ckpt.Checkpointer(
      partial_initializer=kd.ckpts.PartialLoader(
          source=kd.ckpts.KauldronSource('/path/to/original/work_unit/'),
          # Mapping params from <original state> -> <new state>
          new_to_old={
              # '<new_path>':            '<source_path>'
              'params/decoder/layers_0': 'params/encoder',
          },
      )
  )

  # If the checkpoint do not exists, the `partial_initializer` is used to
  # initialize the weights
  init_state = checkpoint.restore(init_state)

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
    new_to_old = {
        k.replace('/', '.'): v.replace('/', '.')
        for k, v in self.new_to_old.items()
    }
    object.__setattr__(self, 'new_to_old', new_to_old)

  @functools.cached_property
  def _new_to_old_path(self) -> dict[path_utils.Path, path_utils.Path]:
    return {
        path_utils.Path.from_str(from_): path_utils.Path.from_str(to)
        for from_, to in self.new_to_old.items()
    }

  def transform(self, state: _T) -> _T:
    """Transform the state by updating it with pre-trained values."""
    # Deep copy (mutable)
    # `orbax` uses nested `dict`, rather than PyTree
    to_state = ocp.utils.serialize_tree(state, keep_empty_nodes=True)

    from_state = self.source.metadata()

    # For each state to copy, make sure the to/from matches
    for to_path, from_path in self._new_to_old_path.items():
      to_state_inner = to_path.get_from(to_state, err_spec=True)
      from_state_inner = from_path.get_from(from_state, err_spec=True)

      # TODO(epot): Restore once flax `save_args_from_target` are removed.
      to_specs = jax.tree_map(lambda x: ..., to_state_inner)
      from_specs = jax.tree_map(lambda x: ..., from_state_inner)
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


@dataclasses.dataclass(frozen=True)
class KauldronSource(CkptSource):
  """Kauldron loader source for `kd.ckpts.PartialLoader`.

  Attributes:
    workdir: The work directory from which the checkpoint should be loaded
    step: Which step to load (default to last one)
  """

  # TODO(epot): Could potentially have more ways to load the checkpoint (e.g.
  # from xid). Or `workdir=kd.xm.workdir_from_xid()` ?
  workdir: epath.PathLike

  _: dataclasses.KW_ONLY

  step: int = -1

  def __post_init__(self):
    object.__setattr__(self, 'workdir', epath.Path(self.workdir))

  @functools.cached_property
  def _ckpt_mgr(self) -> checkpointer.Checkpointer:
    return checkpointer.Checkpointer(
        workdir=self.workdir,
        save_interval_steps=1,
    )

  def metadata(self) -> Any:
    m = self._ckpt_mgr.item_metadata(self.step)
    if m is None:
      # DEPRECATED LEGACY checkpoints
      # Legacy checkpoints save the flatten structure, so cannot be loaded
      raise NotImplementedError(
          f'Partial loading from old checkpoint not supported: {self.workdir}.'
      )
    return m

  def restore(self, item) -> Any:
    """Loads the params from the checkpoint."""

    state = self._ckpt_mgr.restore(
        step=self.step,
        # Use `_NOT_RESTORED` sentinel value as `orbax` will silently
        # forward the additional values not present in the checkpoint.
        initial_state=jax.tree_map(lambda _: _NOT_RESTORED, item),
        restore_kwargs=dict(
            restore_args=orbax_utils.restore_args_from_target(item),
            # Set `transforms={}` to indicate `orbax` to drop the keys not
            # specified in `item`
            transforms={},
        ),
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

  jax.tree_map(_check_is_restored, state)
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
