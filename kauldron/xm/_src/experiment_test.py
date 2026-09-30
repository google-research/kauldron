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

import dataclasses
from unittest import mock

from absl import flags
from kauldron import kxm
import pytest
from xmanager import resource_selector as rs
from xmanager import xm
from xmanager import xm_abc
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
          'samples_per_plugin': 'images=0',
      },
  )
  assert xp.add_tensorboard_borg
  assert xp.add_tensorboard_corp
  assert xp.resolved_tensorboard_args['samples_per_plugin'] == 'images=0'

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
  assert borg_kwargs['args'] == {'samples_per_plugin': 'images=0'}

  mock_corp.assert_called_once()
  _, corp_kwargs = mock_corp.call_args
  assert 'hparams' in corp_kwargs['args']
  assert 'samples_per_plugin' not in corp_kwargs['args']


def test_launch_with_tensorboard_borg_user():
  with mock.patch.object(
      rs,
      'select',
      return_value=[xm.JobRequirements(location='ab')],
      autospec=True,
  ):
    xp = kxm.Experiment(
        jobs={
            'train': kxm.Job(
                target='//my/target',
                platform='jf=2x2',
                cell='ab',
            ),
        },
        cell='ab',
        root_dir='/tmp/some/{cell}/path/to/{author}/',
        executor=xm_abc.Borg(
            borg_user='custom-borg-user',
            autopilot_params=xm_abc.AutopilotParams(
                enabled=True, fixed_replicas=True
            ),
            restricted_credentials=xm_abc.RestrictedCredentials(
                mode=xm_abc.RestrictedCredentials.Mode.ENFORCED
            ),
        ),
        add_tensorboard_borg=True,
        add_tensorboard_corp=True,
    )
    assert xp.resolved_tensorboard_executor is not None
    assert xp.resolved_tensorboard_executor.borg_user == 'custom-borg-user'
    assert xp.resolved_tensorboard_executor.requirements.location == 'ab'
    assert xp.resolved_tensorboard_executor.autopilot_params.enabled
    assert (
        xp.resolved_tensorboard_executor.restricted_credentials
        == xm_abc.RestrictedCredentials(
            mode=xm_abc.RestrictedCredentials.Mode.ENFORCED
        )
    )

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
  assert borg_kwargs['executor'].borg_user == 'custom-borg-user'
  assert borg_kwargs['executor'].requirements.location == 'ab'
  assert borg_kwargs['executor'].autopilot_params.enabled
  assert borg_kwargs[
      'executor'
  ].restricted_credentials == xm_abc.RestrictedCredentials(
      mode=xm_abc.RestrictedCredentials.Mode.ENFORCED
  )

  mock_corp.assert_called_once()
  _, corp_kwargs = mock_corp.call_args
  assert corp_kwargs['executor'].borg_user == 'custom-borg-user'
  assert corp_kwargs['executor'].requirements.location == 'ab'
  assert corp_kwargs['executor'].autopilot_params.enabled


def test_tensorboard_executor_borg_user_precedence():
  mock_flag = mock.MagicMock()
  mock_flag.using_default_value = False
  mock_flag.value = 'flag-borg-user'

  with (
      mock.patch.object(
          rs,
          'select',
          return_value=[xm.JobRequirements(location='ab')],
          autospec=True,
      ),
      mock.patch.dict(flags.FLAGS.__dict__['__flags'], {'borguser': mock_flag}),
      mock.patch.object(flags.FlagValues, 'is_parsed', return_value=True),
  ):
    xp = kxm.Experiment(
        jobs={
            'train': kxm.Job(
                target='//my/target',
                cell='ab',
            ),
        },
        executor=xm_abc.Borg(borg_user='custom-borg-user'),
        tensorboard_executor=xm_abc.Borg(borg_user='tb-specific-user'),
    )
    assert xp.resolved_tensorboard_executor is not None
    assert xp.resolved_tensorboard_executor.borg_user == 'tb-specific-user'


def test_tensorboard_executor_from_job_executor():
  with mock.patch.object(
      rs,
      'select',
      return_value=[xm.JobRequirements(location='ab')],
      autospec=True,
  ):
    xp = kxm.Experiment(
        jobs={
            'train': kxm.Job(
                target='//my/target',
                cell='ab',
                executor=xm_abc.Borg(
                    borg_user='job-borg-user',
                    autopilot_params=xm_abc.AutopilotParams(
                        enabled=True, fixed_replicas=True
                    ),
                    restricted_credentials=xm_abc.RestrictedCredentials(
                        mode=xm_abc.RestrictedCredentials.Mode.DRY_RUN
                    ),
                ),
            ),
        },
    )
    assert xp.resolved_tensorboard_executor is not None
    assert xp.resolved_tensorboard_executor.borg_user == 'job-borg-user'
    assert xp.resolved_tensorboard_executor.requirements.location == 'ab'
    assert xp.resolved_tensorboard_executor.autopilot_params.enabled
    assert (
        xp.resolved_tensorboard_executor.restricted_credentials
        == xm_abc.RestrictedCredentials(
            mode=xm_abc.RestrictedCredentials.Mode.DRY_RUN
        )
    )


def test_main_job_empty_jobs_raises():
  with mock.patch.object(
      rs,
      'select',
      return_value=[],
      autospec=True,
  ):
    xp = kxm.Experiment()
    with pytest.raises(ValueError, match='Experiment has no jobs configured.'):
      _ = xp.main_job


def test_resolved_jobs_custom_subclass():
  @dataclasses.dataclass(frozen=True, kw_only=True)
  class CustomJob(kxm.Job):
    min_hbm: int = 16
    is_fungible: bool = True

  xp = kxm.Experiment(
      jobs={
          'train': CustomJob(
              target='//path/to/my:target',
              platform='jf=2x2',
              min_hbm=32,
          ),
      },
      cell='jn',
      root_dir='/tmp/some/{cell}/path/to/{author}/',
  )
  resolved = xp.resolved_jobs['train']
  assert isinstance(resolved, CustomJob)
  assert resolved.cell == 'jn'
  assert resolved.platform == 'jf=2x2'
  assert resolved.min_hbm == 32
  assert resolved.is_fungible
  assert not hasattr(resolved, 'root_dir')
  xp.launch()
