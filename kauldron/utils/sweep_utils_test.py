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

"""Test."""

import json

from kauldron import kd
from examples import mnist_autoencoder
from kauldron.utils import sweep_utils
from kauldron.xm._src import kauldron_utils
from kauldron.xm._src import sweep_cfg_utils

with kd.konfig.imports():
  from flax import linen as nn  # pylint: disable=g-import-not-at-top


def sweep():
  for bs in [16, 32]:
    yield {
        'eval_ds.batch_size': bs,
        'train_ds.batch_size': bs,
    }


def sweep_model():
  for m in [
      nn.Dense(12),
      nn.Sequential([]),
  ]:
    yield {'model': m}


def test_sweep():
  from kauldron.utils import sweep_utils_test as config_module  # pylint: disable=g-import-not-at-top

  all_sweep_info = list(
      sweep_cfg_utils._sweeps_from_module(
          module=config_module, names=['model', '']
      )
  )
  assert len(all_sweep_info) == 4  # Cross product

  sweep0 = kauldron_utils._encode_sweep_item(all_sweep_info[0])
  assert json.loads(sweep0.job_kwargs[sweep_utils._FLAG_NAME]) == {
      'eval_ds.batch_size': 16,
      'train_ds.batch_size': 16,
      'model': {'__qualname__': 'flax.linen:Dense', '0': 12},
  }


def test_sweep_overwrite():
  assert sweep_utils._FLAG_NAME == kauldron_utils.SWEEP_FLAG_NAME

  cfg = mnist_autoencoder.get_config()
  cfg = sweep_utils.update_with_sweep(  # pytype: disable=wrong-arg-types
      config=cfg,
      sweep_kwargs=json.dumps({
          'seed': 12,
          'train_ds.name': 'imagenet',
          'train_ds.transforms[0].keep[0]': 'other_image',
          'model': {'__qualname__': 'flax.linen:Dense', '0': 12},
      }),
  )
  assert cfg.seed == 12
  assert cfg.train_ds.transforms[0].keep == ['other_image']
  assert cfg.train_ds.name == 'imagenet'
  assert cfg.model == nn.Dense(12)
