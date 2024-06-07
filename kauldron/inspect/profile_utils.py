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

"""Profiling utils."""

from __future__ import annotations

import dataclasses
import functools
import os

from clu import periodic_actions
from etils import epath
from etils import epy
import jax
from kauldron.utils import config_util
import tqdm


class NoopProfiler(config_util.UpdateFromRootCfg):
  """No-op profiler."""

  def __call__(self, step: int):
    return None


@dataclasses.dataclass(frozen=True, kw_only=True)
class Profiler(config_util.UpdateFromRootCfg):
  """`kd.inspect.Profiler`.

  Attributes:
    num_profile_steps: NOTE: No effect on multi-host
    profile_duration_ms: Duration of the profiling
    first_profile: Trigger profiling at step `x`
    all_host: Whether to profile all hosts. This make the profiling slower to
      load in xprof UI. `all_host` only works with `profile_duration_ms`.
    every_steps: Trigger profiling every `x` steps
    every_secs: Trigger profiling every `x` secs
    on_colab: Whether to profile on Colab
    workdir: Automatically set
  """

  # What to profile
  num_profile_steps: int = 5
  profile_duration_ms: int = 3_000
  all_host: bool = False

  # When to profile
  first_profile: int | None = 10
  every_steps: int | None = None
  every_secs: float | None = 3600.0
  on_colab: bool = False

  workdir: epath.Path = config_util.ROOT_CFG_REF.workdir

  @functools.cached_property
  def _profile(self):
    if epy.is_notebook() and not self.on_colab:
      return lambda step: None  # No-op
    elif not self.all_host or jax.process_count() == 1:  # Single process
      return _Profile(
          num_profile_steps=self.num_profile_steps,
          profile_duration_ms=self.profile_duration_ms,
          first_profile=self.first_profile,
          every_steps=self.every_steps,
          every_secs=self.every_secs,
          logdir=self.workdir,
      )
    else:  # Multi-process
      return periodic_actions.ProfileAllHosts(
          logdir=os.fspath(self.workdir),
          profile_duration_ms=self.profile_duration_ms,
          first_profile=self.first_profile,
          every_steps=self.every_steps,
          every_secs=self.every_secs,
      )

  def __call__(self, step: int):
    return self._profile(step)  # pylint: disable=not-callable


class _Profile(periodic_actions.Profile):
  """Wrapper around `clu.periodic_actions.Profile` that display on Colab."""

  def _end_session(self, url: str | None):
    if epy.is_notebook():
      tqdm.tqdm.write(f'Profile at step {self._previous_step}: {url}')
    return super()._end_session(url)
