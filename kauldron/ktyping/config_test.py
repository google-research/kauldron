# Copyright 2025 The kauldron Authors.
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

import kauldron.ktyping as kt
from kauldron.ktyping import config
from kauldron.ktyping import utils


def test_config_update():
  c1 = config.Config(
      typechecking_enabled=True,
      jaxtyping_annotations=config.ReportingPolicy.ERROR,
  )
  c2 = config.Config(typechecking_enabled=False)
  c3 = c1.update(c2)
  assert not c3.typechecking_enabled
  assert c3.jaxtyping_annotations == config.ReportingPolicy.ERROR

  c4 = config.Config(jaxtyping_annotations=config.ReportingPolicy.WARN)
  c5 = c3.update(c4)
  assert not c5.typechecking_enabled
  assert c5.jaxtyping_annotations == config.ReportingPolicy.WARN


def test_get_config_default():
  assert config.get_config() == config.CONFIG


def test_config_context_manager():
  assert config.get_config().typechecking_enabled
  with config.Config(typechecking_enabled=False):
    assert not config.get_config().typechecking_enabled
  assert config.get_config().typechecking_enabled


def test_modify_global_config(monkeypatch):
  assert hasattr(kt, "CONFIG")

  # modify global config
  assert kt.CONFIG.typechecking_enabled
  monkeypatch.setattr(kt.CONFIG, "typechecking_enabled", False)
  assert not config.get_config().typechecking_enabled


def test_add_config_override():
  # modify config override

  cfg_id = kt.add_config_override(
      r"kauldron\.ktyping", config.Config(typechecking_enabled=False)
  )
  try:
    assert config.get_config().typechecking_enabled
    assert not config.get_config(
        utils.CodeLocation(
            description="test",
            file="test.py",
            line=1,
            module_name="kauldron.ktyping",
        )
    ).typechecking_enabled
  finally:
    kt.remove_config_override(cfg_id)
