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
from kauldron import kd
import numpy as np
import orbax.checkpoint as ocp


@flax.struct.dataclass
class A:
  a: Any


def test_checkpoint(tmp_path: pathlib.Path):
  obj = A({
      'a': A(a=np.arange(8)),
      'b': {
          'c': 42,
          'd': np.arange(16),
      },
  })
  ckpt_mgr = ocp.CheckpointManager(
      tmp_path / 'checkpoints',
      kd.ckpts.PyTreeCheckpointer(),
  )

  ckpt_mgr.save(0, obj)

  new_obj = ckpt_mgr.restore(0)
  chex.assert_trees_all_close(obj, new_obj)
