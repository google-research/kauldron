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

"""Pytest plugin to mock XManager."""

# Could move this to `etils.exm` ?

from unittest import mock

import pytest
from xmanager import resource_selector as rs
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_mock

from unittest import mock as _mock ; xmanager_api = _mock.Mock()


class _MockExperiment(xm_mock.MockExperiment):

  # TODO(epot): This should go natively in XManager
  def set_importance(self, importance: xm.Importance):
    del importance
    return None


@pytest.fixture(autouse=True)
def mock_xm():
  """Mock XManager."""
  xm_xp = _MockExperiment()
  with (
      mock.patch.object(
          xm_abc, 'create_experiment', return_value=xm_xp, autospec=True
      ),
      mock.patch.object(
          xm_abc, 'get_current_experiment', return_value=xm_xp, autospec=True
      ),
      mock.patch.object(
          xm_abc,
          'get_current_work_unit',
          lambda: xm_xp._create_experiment_unit({}).result(),  # pytype: disable=attribute-error  # pylint: disable=attribute-error,protected-access
      ),
      mock.patch.object(
          rs,
          'select',
          lambda *jobs: [j.requirements for j in jobs],
      ),
      mock.patch.object(xmanager_api, 'XManagerApi', autospec=True),
  ):
    yield
