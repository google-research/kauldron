# Copyright 2025 The kauldron Authors.
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

_StrDict = MutableMapping[str, str]
FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict
_T = TypeVar('_T')


# TODO(epot): rename to `InitTransform`
class AbstractPartialLoader(abc.ABC):
  """Abstract class for partial checkpoint loaders.

  During state initialization, order is as follow:

  1. Initialize the model params (`model.init()`)
  2. Apply the `init_transform.transform()` to the state
  3. Initialize the optimizer
  4. Apply the `init_transform.transform_after_optimizer()` to the state

  This order allows:

  * To have the optimizer depend on the pre-trained values (e.g. when using
    optax `decay_to_init` and `ema_weight_wrapper` transforms).
  * To restore the optimizer state from a pre-trained checkpoint.
  """

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

  def transform_after_optimizer(self, state: _T) -> _T:
    """Transformation applied after the optimizer has been restored.

    The `transform` method is called before the optimizer has been restored.
    This allows the optimizer to depend on the pre-trained values (e.g. when
    using optax `decay_to_init` and `ema_weight_wrapper` transforms).

    However sometimes, optimizer state also need to be restored from a
    pre-trained checkpoint. This can be done by this method.

    Args:
      state: The `state` object to transform

    Returns:
      The updated `state`
    """
    # No-op by default.
    return state


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

  def transform_after_optimizer(self, state: _T) -> _T:
    for k, tr in self._transforms.items():
      try:
        state = tr.transform_after_optimizer(state)
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
  new_to_old: _StrDict = dataclasses.field(
      default_factory=lambda: FrozenDict({
          'params': 'params',
          'collections': 'collections',
      })
  )
  step: int = -1

  def transform(self, state: _T) -> _T:
    # Before the optimizer, restore everything except the `opt_state` (i.e.,
    # params, collections,...)
    new_to_old, _ = _split_new_to_old(self.new_to_old)
    return self._partial_restore(state, new_to_old)

  def transform_after_optimizer(self, state: _T) -> _T:
    _, new_to_old = _split_new_to_old(self.new_to_old)
    if not new_to_old:  # No need to restore anything.
      return state

    # After the optimizer is initialized, only restore the `opt_state`
    return self._partial_restore(state, new_to_old)

  def _partial_restore(
      self,
      state: _T,
      new_to_old: _StrDict,
  ) -> _T:
    # This could potentially be extended to non-Kauldron checkpoints (like
    # `kd.ckpts.PartialOrbaxLoader`)
    return self._ckpt_mgr.restore(
        _PartialRestoreCheckpointItem(state, new_to_old=new_to_old),
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
  new_to_old: _StrDict

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
      new_partial_restore = ocp.utils.serialize_tree(
          partial_restore[new_path], keep_empty_nodes=True
      )
      kontext.set_by_path(state, new_path, new_partial_restore)

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


def _split_new_to_old(new_to_old: _StrDict) -> tuple[_StrDict, _StrDict]:
  """Split the `new_to_old` mapping between non-`opt_state` and `opt_state`."""
  new_to_old_other = {}
  new_to_old_opt_state = {}
  for new_path, old_path in new_to_old.items():
    if new_path == 'opt_state' or new_path.startswith('opt_state.'):
      new_to_old_opt_state[new_path] = old_path
    else:
      new_to_old_other[new_path] = old_path
  return new_to_old_other, new_to_old_opt_state


# TODO(epot): Deprecate the alias and migrate everyone to
# `kd.from_xid.get_workdir`
def workdir_from_xid(xid: int, wid: int = 1) -> epath.Path:
  return from_xid.get_workdir(xid=xid, wid=wid)
