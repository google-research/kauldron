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

import pathlib
import sys
import textwrap
import types
from typing import Any
from unittest import mock

from absl import flags
from absl.testing import flagsaver
from kauldron import konfig
import pytest


def _write_cfg(path: pathlib.Path, code: str) -> pathlib.Path:
  path.write_text(textwrap.dedent(code).lstrip())
  return path


def _define_and_parse(
    argv: list[str],
    *,
    name: str = "cfg",
    **kwargs: Any,
) -> tuple[flags.FlagValues, flags.FlagHolder[Any]]:
  flag_values = flags.FlagValues()
  with mock.patch.object(sys, "argv", argv):
    cfg_flag = konfig.DEFINE_config_file(
        name,
        None,
        "Config file.",
        flag_values=flag_values,
        **kwargs,
    )
    flag_values(argv)
  return flag_values, cfg_flag


def test_default_none_returns_none_before_and_after_parse() -> None:
  flag_values = flags.FlagValues()
  cfg_flag = konfig.DEFINE_config_file(
      "cfg", None, "Config file.", flag_values=flag_values
  )
  assert flag_values["cfg"].value is None
  flag_values(["prog"])
  assert cfg_flag.value is None


def test_flagsaver_restores_none_and_loaded_config(
    tmp_path: pathlib.Path,
) -> None:
  cfg_path = _write_cfg(
      tmp_path / "sample_cfg.py",
      """
      from kauldron import konfig
      def get_config():
        return konfig.ConfigDict({'seed': 7})
      """,
  )

  flag_values, cfg_flag = _define_and_parse(["prog"])
  assert cfg_flag.value is None
  saved_none = flagsaver.save_flag_values(flag_values)
  with mock.patch.object(sys, "argv", ["prog", f"--cfg={cfg_path}"]):
    flag_values(["prog", f"--cfg={cfg_path}"])
  assert cfg_flag.value == konfig.ConfigDict({"seed": 7})
  flagsaver.restore_flag_values(saved_none, flag_values)
  assert cfg_flag.value is None

  pop_values, pop_flag = _define_and_parse(
      ["prog", f"--cfg={cfg_path}", "--cfg.seed=7"]
  )
  saved_pop = flagsaver.save_flag_values(pop_values)
  pop_values(["prog", "--cfg.seed=99"])
  assert pop_flag.value == konfig.ConfigDict({"seed": 99})
  flagsaver.restore_flag_values(saved_pop, pop_values)
  with pytest.warns(UserWarning, match="not a ModuleConfigDict"):
    assert pop_flag.value == konfig.ConfigDict({"seed": 7})


def test_config_overrides_py_flag_and_cli_str_arg(
    tmp_path: pathlib.Path,
) -> None:
  cfg_path = _write_cfg(
      tmp_path / "param_cfg.py",
      """
      import dataclasses
      from kauldron import konfig

      @dataclasses.dataclass(frozen=True, kw_only=True)
      class ConfigArgs:
        scale: int = 2

      def get_config(args: ConfigArgs = ConfigArgs()):
        return konfig.ConfigDict({
            'scale': args.scale,
            'seed': 1,
            'obj': None,
        })
      """,
  )
  _, cfg_flag = _define_and_parse([
      "prog",
      f"--cfg={cfg_path}:scale=5",
      "--cfg.seed=99",
      "--cfg.obj=py::types.SimpleNamespace(x=7)",
  ])
  resolved = konfig.resolve(cfg_flag.value)
  assert resolved is not None
  assert resolved.scale == 5
  assert resolved.seed == 99
  assert resolved.obj == types.SimpleNamespace(x=7)


def test_get_config_returning_none_raises(tmp_path: pathlib.Path) -> None:
  cfg_path = _write_cfg(
      tmp_path / "none_cfg.py",
      """
      def get_config():
        return None
      """,
  )
  _, cfg_flag = _define_and_parse(["prog", f"--cfg={cfg_path}"])
  with pytest.raises(ValueError, match="returned None"):
    _ = cfg_flag.value


def test_disclaim_key_flags_attributes_to_caller() -> None:
  flag_values = flags.FlagValues()
  konfig.DEFINE_config_file(
      "cfg", None, "Config file.", flag_values=flag_values
  )
  assert flag_values.find_module_defining_flag("cfg") == __name__
  assert "--cfg:" in flag_values.module_help(__name__)


def test_use_legacy_flag_and_lock_config(tmp_path: pathlib.Path) -> None:
  with pytest.raises(AssertionError, match="lock_config is not supported"):
    konfig.DEFINE_config_file(
        "bad", None, flag_values=flags.FlagValues(), lock_config=True
    )

  cfg_path = _write_cfg(
      tmp_path / "legacy_cfg.py",
      """
      from kauldron import konfig
      def get_config():
        return konfig.ConfigDict({'x': 3})
      """,
  )
  _, legacy_flag = _define_and_parse(
      ["prog", f"--cfg={cfg_path}"], use_legacy_flag=True
  )
  assert legacy_flag.value == konfig.ConfigDict({"x": 3})


def test_empty_config_path_raises_io_error() -> None:
  with pytest.raises(OSError, match="empty string"):
    _define_and_parse(["prog", "--cfg="])
