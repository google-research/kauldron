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
from typing import Any
from unittest import mock

from kauldron import konfig
from kauldron.cli import data
import numpy as np


@dataclasses.dataclass
class FakeSpec:
  dtype: Any
  shape: tuple[int, ...]


def test_element_spec(capsys):
  # TODO(klausg): this mocking here is ugly. Write a proper test.
  trainer = dataclasses.make_dataclass("Trainer", ["train_ds"])(
      train_ds=dataclasses.make_dataclass("DS", ["element_spec"])(
          element_spec={
              "image": FakeSpec(dtype=np.float32, shape=(8, 32, 32, 3)),
              "label": FakeSpec(dtype=np.int32, shape=(8,)),
          }
      )
  )
  cfg = konfig.ConfigDict({"seed": 0, "num_train_steps": 1000})
  with mock.patch("kauldron.konfig.resolve", return_value=trainer):
    cmd = data.ElementSpec(cfg=cfg)
    cmd()
  result = capsys.readouterr().out

  assert "image" in result
  assert "float32" in result
  assert "label" in result
  assert "int32" in result


def test_batch(capsys):
  class FakeDS:

    def __iter__(self):
      yield {"image": np.zeros((2, 32, 32, 3))}

  class FakeDataFrame:

    def to_string(self, **_):
      return "Fake stats"

  trainer = dataclasses.make_dataclass("Trainer", ["train_ds"])(
      train_ds=FakeDS()
  )
  cfg = konfig.ConfigDict({"seed": 0})

  with (
      mock.patch("kauldron.konfig.resolve", return_value=trainer),
      mock.patch(
          "kauldron.inspect.get_batch_stats", return_value=FakeDataFrame()
      ),
  ):
    cmd = data.Batch(cfg=cfg)
    cmd()

  result = capsys.readouterr().out
  assert "Fake stats" in result
