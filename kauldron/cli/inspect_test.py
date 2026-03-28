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

import dataclasses
from unittest import mock

from kauldron import konfig
from kauldron.cli import inspect_cli
import numpy as np


def test_model_overview(capsys):
  class FakeDS:

    def __iter__(self):
      yield {"image": np.zeros((2, 32, 32, 3))}

  class FakeDataFrame:

    def to_string(self, **_):
      return "Fake model overview"

  trainer = dataclasses.make_dataclass(
      "Trainer", ["model", "train_ds", "sharding", "raw_cfg", "rng_streams"]
  )(
      model=None,
      train_ds=FakeDS(),
      sharding=dataclasses.make_dataclass("Sharding", ["batch"])(batch=None),
      raw_cfg=None,
      rng_streams=dataclasses.make_dataclass("RngStreams", ["init_rngs"])(
          init_rngs=lambda: {}
      ),
  )
  cfg = konfig.ConfigDict({"seed": 0})

  with (
      mock.patch("kauldron.konfig.resolve", return_value=trainer),
      mock.patch(
          "kauldron.inspect.get_colab_model_overview",
          return_value=FakeDataFrame(),
      ),
  ):
    cmd = inspect_cli.ModelOverview(cfg=cfg)
    cmd()

  result = capsys.readouterr().out
  assert "Fake model overview" in result
