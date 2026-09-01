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

"""Launcher test."""

from unittest import mock
from kauldron import kxm
from xmanager.contrib.internal import tensorboard

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


def test_launch_with_tensorboard():
  xp = kxm.Experiment(
      jobs={
          'train': kxm.Job(
              target='//path/to/my:target',
              platform='jf=2x2',
          ),
      },
      cell='jn',
      root_dir='/tmp/some/{cell}/path/to/{author}/',
      add_tensorboard_borg=True,
      add_tensorboard_corp=True,
      tensorboard_args={
          'samples_per_plugin': 'images=1000',
      },
      tensorboard_corp_args={
          'min_secs_before_update_images': 0,
      },
  )
  assert xp.add_tensorboard_borg
  assert xp.add_tensorboard_corp
  assert xp.resolved_tensorboard_args['samples_per_plugin'] == 'images=1000'
  assert xp.resolved_tensorboard_corp_args['min_secs_before_update_images'] == 0

  with (
      mock.patch.object(
          tensorboard,
          'add_tensorboard_borg',
          autospec=True,
      ) as mock_borg,
      mock.patch.object(
          tensorboard,
          'add_tensorboard_corp',
          autospec=True,
      ) as mock_corp,
  ):
    xp.launch()

  mock_borg.assert_called_once()
  _, borg_kwargs = mock_borg.call_args
  assert borg_kwargs['args'] == {'samples_per_plugin': 'images=1000'}

  mock_corp.assert_called_once()
  _, corp_kwargs = mock_corp.call_args
  assert 'hparams' in corp_kwargs['args']
  assert corp_kwargs['args']['min_secs_before_update_images'] == 0
  assert 'samples_per_plugin' not in corp_kwargs['args']
