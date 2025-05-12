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

"""Unit tests for the Millstone orchestrator."""

from absl.testing import absltest
from kauldron import kxm
from kauldron.contrib.millstone import millstone_orchestrator
from kauldron.xm._src import dir_utils
from xmanager import xm
from xmanager import xm_abc


class MillstoneOrchestratorTest(absltest.TestCase):
  """Unit tests for the Millstone orchestrator."""

  def setUp(self):
    super().setUp()
    self._mock_experiment = self._make_mock_experiment()
    self._mock_work_unit = self._make_mock_work_unit()
    self._mock_get_current_experiment = self.enter_context(
        absltest.mock.patch.object(
            xm_abc,
            "get_current_experiment",
            autospec=True,
            return_value=self._mock_experiment,
        )
    )
    self._mock_trainer_with_millstone = self.enter_context(
        absltest.mock.patch.object(
            xm_millstone,
            "trainer_with_millstone",
            autospec=True,
        )
    )

  def _make_mock_experiment(self) -> absltest.mock.MagicMock:
    """Creates a mock experiment that adds jobs to the mock work unit."""

    mock_xp = absltest.mock.create_autospec(xm.Experiment, instance=True)

    def launch(launch_work_unit, **kwargs):
      """Stores the result in the experiment which must be awaited by caller."""
      result = launch_work_unit(self._mock_work_unit)
      mock_xp.configure_mock(result=result)

    mock_xp.add.side_effect = launch
    return mock_xp

  def _make_mock_work_unit(self) -> absltest.mock.MagicMock:
    """Creates a mock work unit that returns the added jobs."""

    mock_wu = absltest.mock.create_autospec(
        xm_abc.XManagerWorkUnit, instance=True
    )
    mock_wu.add.side_effect = lambda job: job
    return mock_wu

  def _make_mock_job(self, name: str) -> absltest.mock.MagicMock:
    """Creates a mock job with the specified name."""

    mock_job = absltest.mock.create_autospec(kxm.Job, instance=True)
    mock_job.make_xm_job.side_effect = lambda **kwargs: xm.Job(
        name=name,
        executable=absltest.mock.MagicMock(),
        executor=absltest.mock.MagicMock(),
    )
    return mock_job

  @xm.run_in_asyncio_loop
  async def test_millstone_orchestrator(self):
    pw_service_config = xm_millstone.pw_service_config(
        platforms=["cpu:1x64", "df:8x8"],
        cell="auto",
        build_target=(
            "//learning/processing/millstone/examples:millstone_pw_server"
        ),
        build_from_citc=True,
    )
    orchestrator = millstone_orchestrator.MillstoneOrchestrator(
        pw_service_config=pw_service_config,
        pw_backend_target_arg_name="pw_backend_target",
        jobs_with_pw_backend="train",
    )
    orchestrator.launch_jobs(
        resolved_jobs={
            "train": self._make_mock_job(name="train"),
        },
        sweep_info=kxm.SimpleSweep([{"batch_size": 32}]),
        dir_builder=dir_utils.DirectoryBuilder(
            unresolved_root_dir=None,
            subdir_format=dir_utils.SubdirFormat(),
        ),
    )

    await self._mock_experiment.result
    self._mock_trainer_with_millstone.assert_called_once()
    mock_args, mock_kwargs = self._mock_trainer_with_millstone.call_args
    trainer_job = [
        job.name for job in xm.job_operators.flatten_jobs(mock_args[0])
    ]
    self.assertEqual(trainer_job, ["train"])
    self.assertEqual(mock_args[1], pw_service_config)
    self.assertEqual(
        mock_kwargs["pw_backend_target_arg_name"], "pw_backend_target"
    )
    self.assertEqual(mock_kwargs["jobs_with_pw_backend"], "train")


if __name__ == "__main__":
  absltest.main()
