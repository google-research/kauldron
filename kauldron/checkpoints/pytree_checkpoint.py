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

import pickle
from typing import Any, Optional

from etils import epath
from flax.training import orbax_utils
import jax
import orbax.checkpoint as ocp

# TODO(epot): In theory, could save metadata only for step 0, but should be
# careful.

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
    super().__init__(use_ocdbt=True, **kwargs)

  async def async_save(
      self, directory: epath.Path, item: Any, *args, **kwargs
  ) -> Any:
    """Constructs a save operation."""
    # Save metadata
    metadata = {
        "tree_def": jax.tree_util.tree_structure(item),
        "version": _CKPT_VERSION,
    }
    metadata_path = directory / self._kd_metadata_filename
    metadata_path.write_bytes(pickle.dumps(metadata))

    # TODO(epot): Should `orbax_utils.restore_args_from_target` be saved ?

    # Save checkpoint
    save_args = orbax_utils.save_args_from_target(item)
    return await super().async_save(
        directory,
        item,
        *args,
        save_args=save_args,
        **kwargs,
    )

  def restore(  # pytype: disable=signature-mismatch
      self, directory: epath.Path, item: Optional[Any] = None, **kwargs
  ) -> Any:
    """Restores the provided item synchronously.

    Args:
      directory: the directory to restore from.
      item: an item with the same structure as that to be restored.
      **kwargs: additional arguments for restore.

    Returns:
      The restored item.
    """
    # Load metadata
    metadata_path = directory / self._kd_metadata_filename

    # Backward compatibility
    if not metadata_path.exists():
      if item is None:
        print(f"Cannot auto-infer structure of {directory}: Old checkpoint.")
      return super().restore(directory, item, **kwargs)

    # Restore metadata
    metadata = pickle.loads(metadata_path.read_bytes())
    if metadata["version"] > _CKPT_VERSION:
      print(
          f"Loading a newer checkpoint {metadata['version']} in an old codebase"
          f" {_CKPT_VERSION}. Expect trouble..."
      )

    # Restore structure
    tree_def = metadata["tree_def"]
    if item is not None:
      item_def = jax.tree_util.tree_structure(item)
      if item_def != tree_def:
        raise ValueError(
            f"Restored tree_def do not match expected tree_def:\n{item_def} !="
            f" {tree_def}"
        )

      # TODO(epot): Not sure what this does
      restore_args = orbax_utils.restore_args_from_target(item)
    else:
      item = tree_def.unflatten([ocp.RestoreArgs] * tree_def.num_leaves)
      restore_args = None

    return super().restore(directory, item, restore_args=restore_args, **kwargs)

  @property
  def _kd_metadata_filename(self) -> str:
    """Kauldron metadata filename."""
    # TODO(epot): If multiple `PyTreeCheckpointHandler` are used, this will
    # collide.
    return "metadata.kd.pkl"
