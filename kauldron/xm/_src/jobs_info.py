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

"""Job builder."""

from __future__ import annotations

import dataclasses
import functools

from kauldron.xm._src import cfg_provider_utils
from kauldron.xm._src import job_lib
from xmanager import xm


@dataclasses.dataclass(frozen=True, kw_only=True)
class JobsProvider:
  """Provide the jobs to run (`dict[str, kxm.Job]`).

  Example of implementations:

  * `kxm.EmptyJobs`: Empty provider (only use `xp.jobs`)
  * `kxm.KauldronJobs`: Extract the jobs to launch from the `config.py`.

  Attributes:
    cfg_provider: (optional) Eventually the config used. Automatically forwarded
      from `kxm.Experiment`.
  """

  cfg_provider: cfg_provider_utils.ConfigProvider = dataclasses.field(  # pytype: disable=annotation-type-mismatch
      default=None, repr=False
  )

  @functools.cached_property
  def jobs(self) -> dict[str, job_lib.Job]:
    """Jobs to launch.

    Jobs returned here will be merged to the ones defined in
    `xm.Experiment.jobs`.
    """
    raise NotImplementedError("Abstract method")

  def experiment_creation(self, xp: xm.Experiment) -> None:
    """Called after the `xm.Experiment` creation.

    Used for additional set-up, like adding custom tags & artifacts.

    Args:
      xp: The XManager experiment to customize
    """
    pass


@dataclasses.dataclass(frozen=True)
class EmptyJobs(JobsProvider):

  @functools.cached_property
  def jobs(self) -> dict[str, job_lib.Job]:
    return {}
