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

"""Test."""

import pathlib
from typing import Any

import chex
import flax
import jax.numpy as jnp
from kauldron import kd
from kauldron.utils import assert_utils
import numpy as np
import orbax.checkpoint as ocp


@flax.struct.dataclass
class A:
  a: Any


def test_checkpoint(tmp_path: pathlib.Path):
  obj = A({
      'a': A(a=np.asarray([1, 2, 3])),
      'b': {
          'c': 42,
          'd': jnp.asarray([4, 5]),
      },
  })
  ckpt_mgr = ocp.CheckpointManager(
      tmp_path / 'checkpoints',
      kd.ckpts.PyTreeCheckpointer(),
  )

  ckpt_mgr.save(0, obj)

  new_obj = ckpt_mgr.restore(0, obj)
  assert isinstance(new_obj, A)
  chex.assert_trees_all_close(obj, new_obj)
  assert_utils.assert_trees_all_same_type(obj, new_obj)

  # TODO(b/295122555): Restore once the orbax bug is fixed
  # new_obj = ckpt_mgr.restore(0)
  # assert isinstance(new_obj, dict)
  # chex.assert_trees_all_close(ocp.utils.serialize_tree(obj), new_obj)
  # assert_utils.assert_trees_all_same_type(obj, new_obj)
