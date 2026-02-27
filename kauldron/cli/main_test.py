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

from unittest import mock

from kauldron.cli import config
from kauldron.cli import data
from kauldron.cli import main as cli_main
from kauldron.cli import patch_config
import pytest


class TestParseFlags:

  def test_config_show(self):
    args = cli_main.flag_parser(["prog", "config", "show"])
    assert isinstance(args.command, config.Config)
    assert isinstance(args.command.sub_command, config.Show)

  def test_config_resolve(self):
    args = cli_main.flag_parser(["prog", "config", "resolve"])
    assert isinstance(args.command, config.Config)
    assert isinstance(args.command.sub_command, config.Resolve)

  def test_data_element_spec(self):
    args = cli_main.flag_parser(["prog", "data", "element_spec"])
    assert isinstance(args.command, data.Data)
    assert isinstance(args.command.sub_command, data.ElementSpec)

  def test_config_override(self, tmp_path):
    cfg_path = tmp_path / "dummy_cfg.py"
    cfg_path.write_text(
        "from kauldron import konfig\n"
        "def get_config():\n"
        "  return konfig.ConfigDict({'seed': 42})\n"
    )

    argv = [
        "prog",
        "config",
        "show",
        f"--cfg={cfg_path}",
        "--cfg.seed=123",
    ]
    with mock.patch("sys.argv", argv):
      args = cli_main.flag_parser(argv)
    assert isinstance(args.command, config.Config)
    assert isinstance(args.command.sub_command, config.Show)
    from absl import flags  # pylint: disable=g-import-not-at-top

    assert flags.FLAGS.cfg.seed == 123

  def test_no_noun_has_no_command(self):
    with pytest.raises(SystemExit):
      cli_main.flag_parser(["prog"])

  def test_unknown_noun_exits(self):
    with pytest.raises(SystemExit):
      cli_main.flag_parser(["prog", "unknown"])

  def test_no_verb_has_no_command(self):
    with pytest.raises(SystemExit):
      cli_main.flag_parser(["prog", "config"])


class TestPatchFlags:

  def test_default_patch_values(self):
    args = cli_main.flag_parser(["prog", "config", "show"])
    assert args.patch.stop_after_steps == 1
    assert args.patch.batch_size == "devices"
    assert args.patch.skip_checkpointer
    assert not args.patch.skip_eval

  def test_patch_applied_to_config(self, tmp_path):
    cfg_path = tmp_path / "dummy_cfg.py"
    cfg_path.write_text(
        "from kauldron import konfig\n"
        "def get_config():\n"
        "  return konfig.ConfigDict({\n"
        "    'stop_after_steps': 100,\n"
        "    'workdir': '/tmp/test',\n"
        "    'train_ds': {'batch_size': 64, 'shuffle_buffer_size': 1000},\n"
        "    'checkpointer': {'save_interval_steps': 10},\n"
        "  })\n"
    )

    argv = [
        "prog",
        "config",
        "show",
        f"--cfg={cfg_path}",
    ]
    with mock.patch("sys.argv", argv):
      cli_main.flag_parser(argv)
    patcher = patch_config.PatchConfig(stop_after_steps=3, batch_size=16)
    cfg, origin = cli_main._get_config(patcher)

    assert cfg.stop_after_steps == 3
    assert cfg.train_ds.batch_size == 16
    assert cfg.checkpointer is None
    assert isinstance(origin, patch_config.ConfigOrigin)
    assert origin.filename is not None
