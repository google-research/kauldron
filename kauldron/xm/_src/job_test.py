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

"""Unit tests for kxm Jobs."""

from typing import Iterator
from unittest import mock

from kauldron.xm._src import dir_utils
from kauldron.xm._src import job_lib
import pytest


@pytest.fixture(autouse=True)
def mock_job() -> Iterator[mock.Mock]:
  with mock.patch.object(
      job_lib.Job, "executable", new_callable=mock.PropertyMock
  ) as mock_executable:
    mock_executable.return_value = mock.MagicMock()
    yield mock_executable


def test_job():
  job = job_lib.Job(
      name="train",
  )
  xm_job = job.make_xm_job(
      sweep_args={},
      dir_builder=dir_utils.DirectoryBuilder(
          unresolved_root_dir=None,
          subdir_format=dir_utils.SubdirFormat(),
      ),
  )
  assert xm_job.name == "train"
