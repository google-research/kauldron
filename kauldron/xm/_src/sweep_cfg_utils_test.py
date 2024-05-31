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

"""Tests."""

from kauldron import konfig
from kauldron import kxm
import pytest

# Register XM mocking
pytest_plugins = ('kauldron.xm._src.mock_xm',)


def test_sweep():
  xp = kxm.Experiment(
      jobs={
          'train': kxm.Job(
              target='//path/to/my:target',
          ),
      },
      sweep=True,
      sweep_info=kxm.SweepFromCfg(),
      add_tensorboard_borg=False,
      add_tensorboard_corp=False,
  )

  # Fail when config is not yet set
  with pytest.raises(ValueError, match='sweep module not set'):
    xp.launch()

  # Add the config info
  xp = xp.replace(
      cfg_provider=kxm.ConfigProvider.from_module(
          'kauldron.xm._src.sweep_cfg_utils_test'
      )
  )
  xp.launch()


def get_config():
  return konfig.ConfigDict()


def sweep():
  for batch_size in [32, 64]:
    yield {'batch_size': batch_size}
