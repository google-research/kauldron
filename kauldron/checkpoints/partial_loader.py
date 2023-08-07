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
from typing import Any, TypeVar

from etils import edc
from etils import epath
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
from kauldron.core import paths as path_utils
from kauldron.train import checkpointer
import orbax.checkpoint as ocp

_T = TypeVar('_T')


@dataclasses.dataclass(frozen=True, kw_only=True)
class PartialLoader:
  """Specify where to load the weights to initialize a model.

  Allow to use pretrained weights from another model.

  Usage:

  ```python
  checkpoint = kd.train.Checkpoint(
      fallback_loader=kd.ckpts.PartialLoader(
          source=kd.ckpts.KauldronSource('/path/to/original/work_unit/'),
          # Mapping params from <original state> -> <new state>
          old_to_new={
              'params.endoder': 'params.decoder.layers_0',
          },
      )
  )

  # If the checkpoint do not exists, the `fallback_loader` is used to initialize
  # the weights
  init_state = checkpoint.restore(init_state)

  # `init_state.params['decoder']['layers_0']` now contain the previous encoder
  # weights
  ```

  Attributes:
    source: From where to load the checkpoint (kauldron, xid, external
      project,...)
    old_to_new: Mapping the pytree to copy from original checkpoint to new train
      state.
  """

  source: CkptSource
  old_to_new: dict[str, str]  # PyTree[types.EllipsisType] = ...

  def transform(self, state: _T) -> _T:
    """Transform the state by updating it with pre-trained values."""
    # `from` and `to` state
    from_state = self.source.restore()
    to_state = ocp.utils.serialize_tree(state)  # Deep copy (mutable)

    # For each state to copy
    for from_, to in self.old_to_new.items():
      from_path = path_utils.Path.from_str(from_)
      to_path = path_utils.Path.from_str(to)

      # Make sure the paths are correct (will raise error messages)
      from_state_inner = from_path.get_from(from_state, err_spec=True)
      _ = to_path.get_from(to_state, err_spec=True)  # Only for error message

      to_state_inner = to_path[:-1].get_from(to_state, err_spec=True)

      from_specs = etree.spec_like(from_state_inner)
      to_specs = etree.spec_like(to_state_inner[to_path[-1]])
      if from_specs != to_specs:
        raise ValueError(
            'Imported module structure do not match:\n'
            f'From:  {epy.Lines.repr(from_specs)}\n'
            f'Target: {epy.Lines.repr(to_specs)}\n'
            f'Diff: {epy.diff_str(from_specs, to_specs)}'
        )

      # Replace the state
      to_state_inner[to_path[-1]] = from_state_inner

    return ocp.utils.deserialize_tree(to_state, target=state)


class CkptSource(abc.ABC):
  """Partial checkpoint loader source. See `kd.ckpts.PartialLoader`."""

  # TODO(epot): Support partial checkpoint loading
  @abc.abstractmethod
  def restore(self) -> Any:
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

  def restore(self) -> Any:
    """Loads the params from the checkpoint."""
    # TODO(epot): Could try to clean this up

    # Create a checkpoint manager only to compute the correct path.
    ckpt_mgr = checkpointer.Checkpointer(
        workdir=self.workdir,
        save_interval_steps=1,
    )
    path = ckpt_mgr._path_for_step(self.step)  # pylint: disable=protected-access
    if not path.exists():
      raise FileNotFoundError(f'Checkpoint {path} does not exist.')

    # Note: This do not pass `restore_args` !

    # TODO(epot): Could only load the exact params to extract, rather than
    # all params
    # TODO(epot): Could also support list of path selector. Require:
    # * Loading the checkpoint structure with `.metadata()`
    # * Flattening all metadata keys
    # * Applying the selector to the flattened metadata keys to resolve the
    #   regex,...
    # * Unflattening the filtered selectors.
    # TODO(epot): Auto-detect kauldron and add `0.params`

    # params_selector = ocp.utils.from_flat_dict(self.old_to_new, sep='.')
    # params_selector = jax.tree_map(lambda x: ..., params_selector)

    ckpt = ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(
            use_ocdbt=True,
            write_tree_metadata=True,
        ),
    )
    state = ckpt.restore(path)

    if isinstance(state, dict):  # New checkpoint
      return state
    else:  # DEPRECATED checkpoints
      # TODO(epot): Delete after users have migrated
      # Backward compatibility: Old `TrainState.__tree_flatten__`
      # structure
      print(
          f'Loading old checkpoint: {path}. Kauldron will drop support for'
          ' those at some point.'
      )
      assert len(state) == 4, 'Unknown checkpoint structure.'
      return {
          k: v
          for k, v in zip(
              ['step', 'params', 'opt_state', 'training_time_hours'], state
          )
      }
