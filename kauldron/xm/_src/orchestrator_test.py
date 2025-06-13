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

"""Unit tests for the Kauldron orchestrator."""

import asyncio
import itertools
from typing import Iterator
from unittest import mock

from kauldron import kxm
from kauldron.xm._src import dir_utils
from kauldron.xm._src import orchestrator as orchestrator_lib
import pytest
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_mock


class MockExperiment(xm_mock.MockExperiment):
  """Mock XM experiment that returns the added jobs."""

  def __init__(self):
    super().__init__()
    self._work_unit = MockWorkUnit()
    self._results = []

  def add(self, launch_work_unit: xm.JobGeneratorType, **kwargs) -> None:
    self._results.append(launch_work_unit(self._work_unit, **kwargs))

  async def flatten_jobs(self) -> list[xm.Job]:
    await asyncio.gather(*self._results)
    return list(
        itertools.chain.from_iterable(
            xm.job_operators.flatten_jobs(jobs) for jobs in self._work_unit.jobs
        )
    )


class MockWorkUnit(mock.MagicMock):
  """Mock XM work unit that returns the added jobs."""

  def __init__(self):
    super().__init__()
    self._jobs = []

  def add(self, job: xm.JobType) -> None:
    self._jobs.append(job)

  @property
  def jobs(self) -> list[xm.JobType]:
    return self._jobs


class MockJob(kxm.Job):

  name: str

  def make_xm_job(self, **kwargs) -> xm.Job:
    return xm.Job(
        name=self.name,
        executable=mock.MagicMock(),
        executor=mock.MagicMock(),
    )


@pytest.fixture(name="mock_experiment")
def _mock_experiment() -> Iterator[MockExperiment]:
  mock_exp = MockExperiment()
  with mock.patch.object(
      xm_abc,
      "get_current_experiment",
      autospec=True,
      return_value=mock_exp,
  ):
    yield mock_exp


@xm.run_in_asyncio_loop
async def test_orchestrator(mock_experiment: MockExperiment):
  orchestrator = orchestrator_lib.SweepOrchestrator()
  orchestrator.launch_jobs(
      resolved_jobs={
          "train": MockJob(name="train"),
      },
      sweep_info=kxm.SimpleSweep([
          {"batch_size": 32},
          {"batch_size": 64},
      ]),
      dir_builder=dir_utils.DirectoryBuilder(
          unresolved_root_dir=None,
          subdir_format=dir_utils.SubdirFormat(),
      ),
  )

  jobs = await mock_experiment.flatten_jobs()
  jobs = [job.name for job in jobs]
  assert jobs == ["train", "train"]
