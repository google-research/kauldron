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

from __future__ import annotations

import tempfile

from kauldron.cli import main as cli_main
from kauldron.cli import run
from examples import mnist_autoencoder
import tensorflow_datasets as tfds


class TestRunIntegration:

  def test_eval_shape_mnist(self):
    cfg = mnist_autoencoder.get_config()
    cfg.train_ds.batch_size = 2
    with tempfile.TemporaryDirectory(prefix='kauldron_test_') as workdir:
      cfg.workdir = workdir
      cmd = run.EvalShape(cfg=cfg)  # pytype: disable=wrong-arg-types
      with tfds.testing.mock_data():
        result = cmd()
    assert 'eval_shape: OK' in result


class TestRunParseFlags:

  def test_run_eval_shape(self):
    args = cli_main.flag_parser(['prog', 'run', 'eval_shape'])
    assert isinstance(args.command, run.Run)
    assert isinstance(args.command.sub_command, run.EvalShape)

  def test_run_mock_tpu(self):
    args = cli_main.flag_parser(['prog', 'run', 'mock_tpu'])
    assert isinstance(args.command, run.Run)
    assert isinstance(args.command.sub_command, run.MockTpu)

  def test_run_mock_tpu_defaults(self):
    args = cli_main.flag_parser(['prog', 'run', 'mock_tpu'])
    sub = args.command.sub_command
    assert isinstance(sub, run.MockTpu)
    assert sub.platform == 'pf=2x2'

  def test_run_mock_tpu_custom_platform(self):
    args = cli_main.flag_parser(
        ['prog', 'run', 'mock_tpu', '--platform', 'vlp=4x4']
    )
    sub = args.command.sub_command
    assert isinstance(sub, run.MockTpu)
    assert sub.platform == 'vlp=4x4'

  def test_run_cpu(self):
    args = cli_main.flag_parser(['prog', 'run', 'cpu'])
    assert isinstance(args.command, run.Run)
    assert isinstance(args.command.sub_command, run.Cpu)


class TestParsePlatform:

  def test_basic(self):
    platform, topology = run._parse_platform('pf=2x2')
    assert platform == 'Pufferfish'
    assert topology == '2x2'

  def test_alias_vlp(self):
    platform, topology = run._parse_platform('vlp=4x4')
    assert platform == 'Viperlite_Pod'
    assert topology == '4x4'

  def test_full_name_passthrough(self):
    platform, topology = run._parse_platform('Ghostfish=2x2x1')
    assert platform == 'Ghostfish'
    assert topology == '2x2x1'

  def test_case_insensitive(self):
    platform, topology = run._parse_platform('PF=2x2')
    assert platform == 'Pufferfish'
    assert topology == '2x2'

  def test_invalid_format(self):
    import pytest  # pylint: disable=g-import-not-at-top

    with pytest.raises(ValueError, match='Expected'):
      run._parse_platform('pf')
