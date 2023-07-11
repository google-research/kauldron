# Copyright 2023 The kauldron Authors.
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

from etils import epath

from kauldron import kd


def test_config():
  cfg = kd.train.Config(unknown_field='aaa')
  assert cfg.unknown_field == 'aaa'
  assert not hasattr(cfg, 'seed')
  cfg = cfg.replace(seed=1)
  assert cfg.seed == 1
  assert cfg.train_losses == {}  # pylint: disable=g-explicit-bool-comparison

  assert cfg.workdir == epath.Path()
  cfg = cfg.replace(workdir='abc')
  assert cfg.workdir == epath.Path('abc')

  assert not hasattr(cfg, 'new_att_direct')
  cfg = cfg.replace(new_att_direct=1)
  assert cfg.new_att_direct == 1

  assert not hasattr(cfg, 'new_att')
  cfg = cfg.replace(new_attr=123)
  assert cfg.new_attr == 123
  assert cfg.new_att_direct == 1
  assert cfg.seed == 1
  assert cfg.workdir == epath.Path('abc')
  assert cfg.unknown_field == 'aaa'
