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

from kauldron import konfig
from kauldron.cli import config


def test_show_prints_repr(capsys):
  cfg = konfig.ConfigDict({"seed": 42, "num_train_steps": 1000})
  cmd = config.Show(cfg=cfg)
  cmd()

  captured = capsys.readouterr().out
  assert "'seed': 42" in captured
  assert "'num_train_steps': 1000" in captured


def test_resolve_prints_config(capsys):
  cfg = konfig.ConfigDict({"seed": 42, "num_train_steps": 1000})
  cmd = config.Resolve(cfg=cfg, verbose=True)
  cmd()

  captured = capsys.readouterr().out
  assert "'seed': 42" in captured
  assert "'num_train_steps': 1000" in captured
