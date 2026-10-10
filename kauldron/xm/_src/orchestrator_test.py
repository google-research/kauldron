# Copyright 2026 The kauldron Authors.
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

from absl import flags
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
    self.identities = []

  def add(self, launch_work_unit: xm.JobGeneratorType, **kwargs) -> None:  # pyrefly: ignore[bad-override]
    self.identities.append(kwargs["identity"])
    self._results.append(launch_work_unit(self._work_unit, **kwargs))

  def add_existing_work_units(self, num_work_units: int) -> None:
    """Adds work units that the experiment has before the launch."""
    for _ in range(num_work_units):
      self._work_units[len(self._work_units) + 1] = MockWorkUnit()

  async def flatten_jobs(self) -> list[xm.Job]:
    await asyncio.gather(*self._results)
    return list(
        itertools.chain.from_iterable(
            xm.job_operators.flatten_jobs(jobs) for jobs in self._work_unit.jobs  # pyrefly: ignore[bad-argument-type]
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


def _launch_sweep() -> None:
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


@xm.run_in_asyncio_loop
async def test_orchestrator(mock_experiment: MockExperiment):
  _launch_sweep()

  jobs = await mock_experiment.flatten_jobs()
  jobs = [job.name for job in jobs]
  assert jobs == ["train", "train"]
  assert mock_experiment.identities == ["sweep_0", "sweep_1"]


@pytest.mark.parametrize(
    "xreload_xid, expected_identities",
    [
        # Adding work units to an existing experiment (`kxm.Experiment.xid`).
        (None, ["sweep_2", "sweep_3"]),
        # XReload re-adds the experiment's own work units.
        (12345, ["sweep_0", "sweep_1"]),
    ],
)
@xm.run_in_asyncio_loop
async def test_orchestrator_with_existing_work_units(
    mock_experiment: MockExperiment,
    xreload_xid: int | None,
    expected_identities: list[str],
):
  mock_experiment.add_existing_work_units(2)
  with mock.patch.dict(
      flags.FLAGS.__dict__["__flags"],
      {"xreload_xid": mock.MagicMock(value=xreload_xid)},
  ):
    _launch_sweep()

  await mock_experiment.flatten_jobs()
  assert mock_experiment.identities == expected_identities
