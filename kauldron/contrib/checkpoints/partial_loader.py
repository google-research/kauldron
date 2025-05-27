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

from collections.abc import MutableMapping
import dataclasses
import typing
from typing import TypeVar

from clu import checkpoint as clu_checkpoint  # pytype: disable=import-error
from etils import epath
import flax
import jax
from kauldron import kontext
from kauldron.checkpoints import partial_loader
import orbax.checkpoint as ocp


FrozenDict = dict if typing.TYPE_CHECKING else flax.core.FrozenDict
_T = TypeVar('_T')


@dataclasses.dataclass(frozen=True, kw_only=True)
class PartialCLULoader(partial_loader.AbstractPartialLoader):
  """Parial loader for CLU checkpoints.

  Allow to use pretrained weights from a CLU checkpoint.

  Usage:

  ```python
  cfg.init_transform = kd.contrib.ckpts.PartialCLULoader(
      workdir='/path/to/original/work_unit/',
      new_to_old={  # Mapping params
          # '<new_path>':            '<source_path>'
          'params.decoder.layers_0': 'params.endoder',
      },
  )

  trainer = konfig.resolve(cfg)

  # When initializing the weights, the `init_transform` is applied
  init_state = trainer.init_state()

  # `init_state.params['decoder']['layers_0']` now contain the previous encoder
  # weights
  ```

  Attributes:
    ckpt_dir: The directory from which the checkpoint should be loaded.
    new_to_old: Mapping the pytree to copy to the new state from the original
      checkpoint. By default, copy all model `params` and `collections`
  """

  ckpt_dir: epath.PathLike
  new_to_old: MutableMapping[str, str] = dataclasses.field(
      default_factory=lambda: FrozenDict({
          'params': 'params',
          'collections': 'collections',
      })
  )

  def transform(self, state: _T) -> _T:
    restored = clu_checkpoint.load_state_dict(self.ckpt_dir)

    # `state` is not a PyTree so we make a mutable deep copy first.
    state_serialized = ocp.utils.serialize_tree(state, keep_empty_nodes=True)

    def get_with_sharding(new_path, old_path):
      """Copy over elements at old_path with the sharding of elements atnew_path."""
      old = kontext.get_by_path(restored, old_path)
      new = kontext.get_by_path(state_serialized, new_path)

      def get_pytree_elem(x, path):
        """Recursively get the element at path in x."""
        try:
          return x[path]
        except KeyError:
          return get_pytree_elem(x[path[0].key], path[1:])

      old_s = jax.tree.map_with_path(
          lambda path, x: jax.device_put(
              x, get_pytree_elem(new, path).sharding
          ),
          old,
      )
      return old_s

    # Extract sub-tree from old state, and shard it according to new state
    sub_state = {
        new_path: get_with_sharding(new_path, old_path)
        for new_path, old_path in self.new_to_old.items()
    }

    # Update keys in `state_serialized` using new values from `sub_state`.
    for new_path, _ in self.new_to_old.items():
      kontext.set_by_path(state_serialized, new_path, sub_state[new_path])

    # Writes back into `state` from the updated `state_serialized`.
    state = ocp.utils.deserialize_tree(
        state_serialized, target=state, keep_empty_nodes=True
    )
    return state
