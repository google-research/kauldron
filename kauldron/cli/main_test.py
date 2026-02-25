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
import pytest


class TestParseFlags:

  def test_config_show(self):
    args = cli_main.flag_parser(["prog", "config", "show"])
    assert isinstance(args.command, config.ConfigCmd)
    assert isinstance(args.command.sub_command, config.Show)

  def test_config_resolve(self):
    args = cli_main.flag_parser(["prog", "config", "resolve"])
    assert isinstance(args.command, config.ConfigCmd)
    assert isinstance(args.command.sub_command, config.Resolve)

  def test_data_element_spec(self):
    args = cli_main.flag_parser(["prog", "data", "element_spec"])
    assert isinstance(args.command, data.DataCmd)
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
    assert isinstance(args.command, config.ConfigCmd)
    assert isinstance(args.command.sub_command, config.Show)
    assert args.command.sub_command.cfg is None
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


class TestMain:

  def test_no_command_raises(self):
    args = mock.Mock(spec=[])
    with pytest.raises(SystemExit, match="No command specified"):
      cli_main.main(args)
