# Copyright 2025 The kauldron Authors.
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

"""Orchestrator for launching Kauldron experiments with Millstone servers.

Example usage:

```
def get_config() -> kxm.Experiment:

  pw_service_config = xm_millstone.pw_service_config(
      platforms=["cpu:1x64", "pf:4x4x4"],
      cell="auto",
      build_target=(
          "//learning/processing/millstone/examples:millstone_pw_server"
      ),
      build_from_citc=True,
  )

  return kxm.Experiment(
      ...
      orchestrator=millstone_orchestrator.MillstoneOrchestrator(
          pw_service_config=pw_service_config,
          pw_backend_target_arg_name="pw_backend_target",
          jobs_with_pw_backend="train",
      ),
      ...
  )
```
"""

from typing import Sequence

from kauldron import kxm
from kauldron.xm._src import dir_utils
from kauldron.xm._src import orchestrator
from xmanager import xm
from xmanager import xm_abc


class MillstoneOrchestrator(orchestrator.SweepOrchestrator):
  """Launch all jobs in a single work-unit with Millstone servers."""

  def __init__(
      self,
      *,
      pw_service_config: service_lib.ServiceConfig,
      pw_backend_target_arg_name: str,
      jobs_with_pw_backend: str | Sequence[str] | None = None,
  ):
    super().__init__()
    self._pw_service_config = pw_service_config
    self._pw_backend_target_arg_name = pw_backend_target_arg_name
    self._jobs_with_pw_backend = jobs_with_pw_backend

  def _create_job_group(
      self,
      wu: xm_abc.XManagerWorkUnit,
      *,
      resolved_jobs: dict[str, kxm.Job],
      sweep_item: kxm.SweepItem,
      dir_builder: dir_utils.DirectoryBuilder,
  ) -> xm.JobGroup:
    """Creates the XManager jobs and Pathways servers."""

    job_group = super()._create_job_group(
        wu=wu,
        resolved_jobs=resolved_jobs,
        sweep_item=sweep_item,
        dir_builder=dir_builder,
    )
    return xm_millstone.trainer_with_millstone(
        job_group,
        self._pw_service_config,
        work_unit=wu,
        pw_backend_target_arg_name=self._pw_backend_target_arg_name,
        jobs_with_pw_backend=self._jobs_with_pw_backend,
    )
