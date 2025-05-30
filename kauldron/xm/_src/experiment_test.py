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

"""Launcher test."""

from kauldron import kxm

# Register XM mocking
pytest_plugins = ('kauldron.xm._src.mock_xm',)


def test_launch_no_workdir():
  xp = kxm.Experiment(
      jobs={
          'train': kxm.Job(
              target='//path/to/my:target',
              platform='jf=2x2',
              args={
                  'batch_size': 128,
              },
          ),
      },
  )
  xp.launch()


def test_launch_workdir():
  xp = kxm.Experiment(
      jobs={
          'train': kxm.Job(
              target='//path/to/my:target',
              args={
                  'workdir': kxm.WU_DIR_PROXY,
              },
              platform='jf=2x2',
          ),
      },
      # Cell has to be provided (as auto-select not available in test)
      cell='jn',
      root_dir='/tmp/some/{cell}/path/to/{author}/',
  )
  xp.launch()

  # TODO(epot): Add more tests (e.g. check the arguments are correctly updated)
