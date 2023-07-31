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

"""Rngs."""

from __future__ import annotations

import dataclasses

import jax
from kauldron import random as kd_random


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class RngInfo:
  """Info on one `rng` stream.

  Attributes:
    name: Stream name
    init: Whether the rng is used in `model.init`
    train: Whether the rng is used in train (`is_training=True`)
    eval: Whether the rng is used in eval (`is_training=False`)
    per_step: Whether the rng is different at each step
    per_process: Whether the rng is different for each process
  """

  name: str

  init: bool = True
  train: bool = True
  eval: bool = False

  per_step: bool = True
  per_process: bool = True

  def make(self, rng: kd_random.PRNGKey, *, step: int) -> kd_random.PRNGKey:
    rng = rng.fold_in(self.name)
    if self.per_step:
      rng = rng.fold_in(step)
    if self.per_process:
      rng = rng.fold_in(jax.process_index())
    return rng


RNGS = [
    RngInfo(
        name='params',
        init=True,
        train=False,
        eval=False,
        per_step=False,
        per_process=False,
    ),
    # TODO(epot): Once users can configure the `rng_info` in their config,
    # eval should be explicitly set (to raise error if rng is
    # used accidentally)
    # TODO(epot): `state_init` seems only used in videosrt. Instead should
    # explicitly set in the videosrt config
    RngInfo(name='state_init', eval=True),
    RngInfo(name='dropout', eval=True),
    RngInfo(name='default', eval=True),
]
