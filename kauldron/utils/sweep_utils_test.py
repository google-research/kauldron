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

import contextlib
import shlex
import sys

from absl import flags
from kauldron import kd
from examples import mnist_autoencoder
from kauldron.xm._src import kauldron_utils
from kauldron.xm._src import sweep_cfg_utils

with kd.konfig.imports():
  # pylint: disable=g-import-not-at-top
  from flax import linen as nn
  from kauldron import data as kd_data
  # pylint: enable=g-import-not-at-top


def sweep():
  for bs in [16, 32]:
    yield {
        'eval_ds.batch_size': bs,
        'train_ds.batch_size': bs,
        'aux.model_size': 'big' if bs == 16 else 'small',
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
  assert sweep0.job_kwargs == {
      'cfg.eval_ds.batch_size': '16',
      'cfg.train_ds.batch_size': '16',
      'cfg.aux.model_size': 'big',
      'cfg.model': '{"__qualname__": "flax.linen:Dense", "0": 12}',
  }
  sweep0 = kauldron_utils.deserialize_job_kwargs(sweep0.job_kwargs)
  assert sweep0 == {
      'eval_ds.batch_size': 16,
      'train_ds.batch_size': 16,
      'aux.model_size': 'big',
      'model': {'__qualname__': 'flax.linen:Dense', '0': 12},
  }


def test_sweep_overwrite():
  argv = shlex.split(
      # fmt: off
      'my_app'
      f' --cfg={mnist_autoencoder.__file__}'
      ' --cfg.seed=12'
      ' --cfg.schedules.none_1=null'  # Json
      ' --cfg.schedules.none_2=None'  # Python
      ' --cfg.schedules.none_3="\'None\'"'  # Python
      ' --cfg.schedules.value_1="{\\"null\\": null, \\"false\\": false, \\"1\\": 1}"'  # Json   # pylint: disable=line-too-long
      ' --cfg.schedules.value_2="{\\"None\\": None, \\"false\\": False, \\"1\\": 1}"'  # Python  # pylint: disable=line-too-long
      ' --cfg.train_ds.name=imagenet'
      ' --cfg.train_ds.transforms[0].keep[0]=other_image'
      ' --cfg.model="{\\"__qualname__\\": \\"flax.linen:Dense\\", \\"0\\": 12}"'
      ' --cfg.evals.eval.ds.transforms="['
      '{\\"__qualname__\\": \\"kauldron.data:ValueRange\\", '
      '\\"key\\": \\"image\\", \\"vrange\\": (0,1)}]"'
      # fmt: on
  )

  flag_values = flags.FlagValues()
  with _replace_sys_argv(argv):
    sweep_flag = kd.konfig.DEFINE_config_file(
        'cfg',
        mnist_autoencoder.__file__,
        'Config file to use for the sweep.',
        flag_values=flag_values,
    )
    flag_values(argv)

  cfg = sweep_flag.value
  assert cfg.seed == 12
  assert cfg.train_ds.transforms[0].keep == ['other_image']
  assert cfg.train_ds.name == 'imagenet'
  assert cfg.model == nn.Dense(12)
  assert cfg.schedules.none_1 is None
  assert cfg.schedules.none_2 is None
  assert cfg.schedules.none_3 == 'None'
  assert cfg.schedules.value_1 == kd.konfig.ConfigDict({
      'null': None,
      'false': False,
      '1': 1,
  })
  assert cfg.schedules.value_2 == kd.konfig.ConfigDict({
      'None': None,
      'false': False,
      '1': 1,
  })
  assert cfg.evals.eval.ds.transforms == [
      kd_data.ValueRange(key='image', vrange=(0, 1))
  ]
  assert isinstance(cfg.model, kd.konfig.ConfigDict)


@contextlib.contextmanager
def _replace_sys_argv(argv):
  old_argv = sys.argv
  sys.argv = argv
  try:
    yield
  finally:
    sys.argv = old_argv
