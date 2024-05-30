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

"""Orchestrator class."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Any

from kauldron.xm._src import dir_utils
from kauldron.xm._src import job_lib
from kauldron.xm._src import sweep_utils
from xmanager import xm
from xmanager import xm_abc


@dataclasses.dataclass(frozen=True)
class Orchestrator(abc.ABC):
  """The role of the Orchestrator is to schedule the jobs.

  For example:

  * Run a hyper parameter sweep
  * Schedule jobs in a specific order (dependency graph)
  * Vizier hyper parameter search

  For most use-cases, the default `kxm.SweepOrchestrator` should work.
  """

  @abc.abstractmethod
  def launch_jobs(
      self,
      *,
      resolved_jobs: dict[str, job_lib.Job],
      sweep_info: sweep_utils.SweepInfo,
      dir_builder: dir_utils.DirectoryBuilder,
  ) -> None:
    """Launch all jobs / work-units."""
    raise NotImplementedError()


class SweepOrchestrator(Orchestrator):
  """Launch all jobs in a single work-unit, eventually with parameter sweep.

  * Launch all jobs in a single work-unit
  * For sweep, all jobs in a work-unit are updated with the same sweep kwargs
  """

  def launch_jobs(
      self,
      *,
      resolved_jobs: dict[str, job_lib.Job],
      sweep_info: sweep_utils.SweepInfo,
      dir_builder: dir_utils.DirectoryBuilder,
  ) -> None:
    xp = xm_abc.get_current_experiment()

    # TODO(klausg): Add a confirmation dialogue before starting lots of workers?
    for i, sweep_item in enumerate(sweep_info):
      xp.add(
          functools.partial(
              self._launch_work_unit,
              resolved_jobs=resolved_jobs,
              sweep_item=sweep_item,
              dir_builder=dir_builder,
          ),
          # Args passed here will surface on the UI
          #
          args=sweep_item.xm_ui_kwargs,
          identity=f"sweep_{i}",
      )

  async def _launch_work_unit(
      self,
      wu: xm_abc.XManagerWorkUnit,
      *,
      resolved_jobs: dict[str, job_lib.Job],
      sweep_item: sweep_utils.SweepItem,
      dir_builder: dir_utils.DirectoryBuilder,
      **xm_ui_sweep_args: Any,
  ) -> None:
    """Launch a single work-unit."""
    # Do not pass `xm_ui_sweep_args` as explicit argument (only used for
    # display).
    del xm_ui_sweep_args

    dir_builder = dir_builder.replace_ctx(
        wu=wu,
        sweep_item=sweep_item,
    )

    xm_jobs = {
        k: job.make_xm_job(
            sweep_args=sweep_item.job_kwargs, dir_builder=dir_builder
        )
        for k, job in resolved_jobs.items()
    }
    wu.add(xm.JobGroup(**xm_jobs))
