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

"""Checkpoint handler that support arbitrary `PyTree`."""

from __future__ import annotations

from typing import Any, Optional, TypeVar

from etils import epath
from flax.training import orbax_utils
import orbax.checkpoint as ocp

# TODO(epot): In theory, could save metadata only for step 0, but should be
# careful.

_T = TypeVar('_T')

_CKPT_VERSION = 1


class PyTreeCheckpointer(ocp.Checkpointer):
  """Low level checkpointer that support arbitrary pytree.

  ```python
  ckpt = kd.train.PyTreeCheckpointer()

  ckpt.save(path, state)
  state = ckpt.restore(path)
  ```
  """

  def __init__(self):
    super().__init__(PyTreeCheckpointHandler())


class PyTreeCheckpointHandler(ocp.PyTreeCheckpointHandler):
  """PyTree checkpoint handler that support arbitrary pytree."""

  def __init__(self, **kwargs):
    # Use OCDBT for faster performance (see
    # https://orbax.readthedocs.io/en/latest/optimized_checkpointing.html)
    super().__init__(
        use_ocdbt=True,
        write_tree_metadata=True,
        **kwargs,
    )

  def restore(  # pytype: disable=signature-mismatch
      self,
      directory: epath.Path,
      item: Optional[_T] = None,
      *,
      restore_args: Optional[Any] = None,
      **kwargs,
  ) -> _T:
    """Restores the provided item synchronously.

    Args:
      directory: the directory to restore from.
      item: an item with the same structure as that to be restored. If missing
        the tree will be restored from the saved structure.
      restore_args: Restore args
      **kwargs: additional arguments for restore.

    Returns:
      The restored item.
    """
    if item is not None and restore_args is None:
      # Restore args make sure the restored arrays match the `item` (
      # jax vs numpy array,...)
      restore_args = orbax_utils.restore_args_from_target(item)
    if item is None and restore_args is None:
      raise ValueError(
          'Please comment on b/295122555 so orbax team prioritise this bug. '
          'Currently cannot restore without `item=` or `restore_args=`.'
      )
    return super().restore(
        directory=directory,
        item=item,
        restore_args=restore_args,
        **kwargs,
    )
