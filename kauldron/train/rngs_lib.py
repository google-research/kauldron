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

"""Rngs."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import functools

import jax
from kauldron import random as kd_random
from kauldron.utils import config_util

Rngs = dict[str, kd_random.PRNGKey]

_jit_method = functools.partial(jax.jit, static_argnames=['self'])


@dataclasses.dataclass(frozen=True, eq=True)
class RngStream:
  """Info on one `rng` stream.

  See doc at https://kauldron.rtfd.io/en/latest/eval.html#rng-streams

  Attributes:
    name: Stream name
    init: Whether the rng is used in `model.init`
    train: Whether the rng is used in train (`is_training=True`)
    eval: Whether the rng is used in eval (`is_training=False`)
    per_step: Whether the rng is different at each step
  """

  name: str

  _: dataclasses.KW_ONLY

  init: bool = True
  train: bool = True
  eval: bool = False

  per_step: bool = True

  def make(
      self,
      rng: kd_random.PRNGKey,
      *,
      step: int | None = None,
      key: str | None = None,
  ) -> kd_random.PRNGKey:
    """Create the `rng` from the global root rng.

    Arguments:
      rng: The root rng, common to all processes
      step: Current model step
      key: Additional string (e.g. `train`, `init`,...) to fold in

    Returns:
      The new rng
    """
    rng = rng.fold_in(self.name)
    if self.per_step:
      self._assert_is_not_none(step, 'step')
      rng = rng.fold_in(step)
    if key is not None:  # Additional key to fold (e.g. `train`, `eval`)
      rng = rng.fold_in(key)
    return rng

  def _assert_is_not_none(self, val, name: str) -> None:
    if val is None:
      raise ValueError(
          f'Missing kwargs `{name}` to generate rng stream: {self}'
      )


_DEFAULT_STREAMS = [
    RngStream(
        'params',
        init=True,
        train=False,
        eval=False,
        per_step=False,
    ),
    RngStream('dropout'),
    RngStream('default'),
]


@dataclasses.dataclass(frozen=True, eq=True)
class RngStreams(config_util.UpdateFromRootCfg):
  """Manager of rng streams.

  See doc at https://kauldron.rtfd.io/en/latest/eval.html#rng-streams

  Generate the `rngs` dict to pass to `model.init` / `model.apply`.

  3 streams are always added: `params`, `dropout`, `default` but their values
  can be overwritten with `stream_overwrites`.

  Attributes:
    stream_overwrites: Additional streams to add. Will be merged with the
      default ones.
    seed: Seed to initialize the root_rng. If `None`, will reuse the global seed
      from `kd.train.Trainer`
  """

  stream_overwrites: Sequence[RngStream] = dataclasses.field(
      default_factory=tuple
  )

  _: dataclasses.KW_ONLY
  seed: int = config_util.ROOT_CFG_REF.seed

  @functools.cached_property
  def streams(self) -> dict[str, RngStream]:
    """Streams (after default are merged)."""
    streams = {s.name: s for s in _DEFAULT_STREAMS}
    streams |= {s.name: s for s in self.stream_overwrites}
    return streams

  # TODO(epot): Benchmark vs having a cached root_rng

  # This is important this function is not cached, because it can be executed
  # in both `@jax.jit` and non-jit contexts.
  @property
  @_jit_method
  def root_rng(self) -> kd_random.PRNGKey:
    """Base root rng from which others are derived."""
    self._assert_root_cfg_resolved()

    if self.seed is None:
      raise ValueError('RngStreams.seed should be set.')
    return kd_random.PRNGKey(self.seed)

  # Could try to unify and have a more flexible mode system (for custom
  # eval/train mode).

  @_jit_method
  def init_rngs(self) -> Rngs:
    """Rngs for `model.init()`."""
    return {  # pylint: disable=g-complex-comprehension
        r.name: r.make(
            self.root_rng,
            step=0,
            key='init',
        )
        for r in self.streams.values()
        if r.init
    }

  @_jit_method
  def train_rngs(self, step: int) -> Rngs:
    """Rngs for `model.apply(..., is_training_property=True)`.

    Args:
      step: Current train/eval step

    Returns:
      rngs: The `dict[<stream name>, kd.random.PRNGKey]`
    """
    return {  # pylint: disable=g-complex-comprehension
        r.name: r.make(
            self.root_rng,
            step=step,
            key='train',
        )
        for r in self.streams.values()
        if r.train
    }

  @_jit_method
  def eval_rngs(self, step: int) -> Rngs:
    """Rngs for `model.apply(..., is_training_property=False)`.

    Args:
      step: Current train/eval step

    Returns:
      rngs: The `dict[<stream name>, kd.random.PRNGKey]`
    """
    return {  # pylint: disable=g-complex-comprehension
        r.name: r.make(
            self.root_rng,
            step=step,
            key='eval',
        )
        for r in self.streams.values()
        if r.eval
    }
