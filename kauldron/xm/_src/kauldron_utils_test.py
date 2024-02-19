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

"""Launcher test."""

from kauldron import konfig
from kauldron.xm.configs import kd_base

# Register XM mocking
pytest_plugins = ('kauldron.xm._src.mock_xm',)


def test_launch():
  xp_cfg = kd_base.get_config()
  xp_cfg.name = 'my_experiment_name'  # Test name conflict
  xp_cfg.cell = 'xx'  # Auto-cell selection not activated in tests
  xp_cfg.jobs_provider.module = 'kauldron.examples.tiny_vit_imagenet'
  xp_cfg.jobs_provider.overrides['xm_job.target'] = (  # pytype: disable=attribute-error
      '//third_party/py/kauldron/examples:trainer'
  )
  with konfig.set_lazy_imported_modules():
    xp = konfig.resolve(xp_cfg)
  xp.launch()


def test_launch_with_evals():
  xp_cfg = kd_base.get_config()
  xp_cfg.cell = 'xx'  # Auto-cell selection not activated in tests
  xp_cfg.jobs_provider.module = (
      'kauldron.examples.mnist_autoencoder_remote_eval'
  )
  xp_cfg.jobs_provider.overrides['xm_job.target'] = (  # pytype: disable=attribute-error
      '//third_party/py/kauldron/examples:trainer'
  )
  with konfig.set_lazy_imported_modules():
    xp = konfig.resolve(xp_cfg)
  xp.launch()


def test_launch_with_shared_evals():
  xp_cfg = kd_base.get_config()
  xp_cfg.cell = 'xx'  # Auto-cell selection not activated in tests
  xp_cfg.jobs_provider.module = (
      'kauldron.examples.mnist_autoencoder_remote_shared_eval'
  )
  xp_cfg.jobs_provider.overrides['xm_job.target'] = (  # pytype: disable=attribute-error
      '//third_party/py/kauldron/examples:trainer'
  )
  with konfig.set_lazy_imported_modules():
    xp = konfig.resolve(xp_cfg)
  xp.launch()
