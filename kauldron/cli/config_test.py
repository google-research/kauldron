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


def test_show_returns_repr():
  """show() should return the repr of the unresolved ConfigDict."""
  cfg = konfig.ConfigDict({"seed": 42, "num_train_steps": 1000})
  result = config.Show(cfg=cfg).execute()

  assert "seed" in result
  assert "num_train_steps" in result


def test_show_with_qualname():
  """show() should include __qualname__ in the repr."""
  cfg = konfig.ConfigDict({
      "__qualname__": "kauldron.train.trainer_lib:Trainer",
      "seed": 0,
      "num_train_steps": 100,
  })
  result = config.Show(cfg=cfg).execute()

  assert "Trainer" in result
  assert "num_train_steps" in result


def test_resolve_simple():
  cfg = konfig.ConfigDict({"seed": 42, "num_train_steps": 1000})
  result = config.Resolve(cfg=cfg).execute()

  assert "seed" in result
  assert "42" in result

