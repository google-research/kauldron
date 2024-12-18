# Copyright 2024 The kauldron Authors.
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

from kauldron import kd
from kauldron.optim import _masks


def test_select():
  # Check the regex is restricted to the exact path.
  assert kd.optim.select("lora")({
      "lora": 0,
      "notlora": 0,
      "lora.more": 0,
      "loranot.more": 0,
      "notlora.more": 0,
      "more.lora": 0,
      "more.notlora": 0,
      "more.lora.more": 0,
      "more.notlora.more": 0,
  }) == {
      "lora": True,
      "notlora": False,
      "lora.more": True,
      "loranot.more": False,
      "notlora.more": False,
      "more.lora": True,
      "more.notlora": False,
      "more.lora.more": True,
      "more.notlora.more": False,
  }

  # Exclude returns the opossite mask.
  assert kd.optim.exclude("lora")({
      "lora": 0,
      "notlora": 0,
      "lora.more": 0,
      "loranot.more": 0,
      "notlora.more": 0,
      "more.lora": 0,
      "more.notlora": 0,
      "more.lora.more": 0,
      "more.notlora.more": 0,
  }) == {
      "lora": False,
      "notlora": True,
      "lora.more": False,
      "loranot.more": True,
      "notlora.more": True,
      "more.lora": False,
      "more.notlora": True,
      "more.lora.more": False,
      "more.notlora.more": True,
  }

  # Test that a `.` in the path is properly escaped.
  assert kd.optim.select("lora.more")({
      "lora": 0,
      "loraxmore": 0,
      "lora.more": 0,
      "more.loraxmore.more": 0,
      "more.lora.more.more": 0,
  }) == {
      "lora": False,
      "loraxmore": False,
      "lora.more": True,
      "more.loraxmore.more": False,
      "more.lora.more.more": True,
  }

  # Test that the select works on nested tree
  assert kd.optim.select("lora.more")({
      "lora": {
          "more": {
              "x": 0,
              "y": 0,
          },
          "notmore": 0,
      },
      "y": {"lora": {"more": 0}},
      "z": 0,
  }) == {
      "lora": {
          "more": {
              "x": True,
              "y": True,
          },
          "notmore": False,
      },
      "y": {"lora": {"more": True}},
      "z": False,
  }

  # Tests that regex are properly escaped
  assert kd.optim.select("lora[0-9]+")({
      "lora00": 0,
      "lora1": 0,
      "lora1x": 0,
      "lora1": 0,
      "xx.lora": 0,
      "xx.lora3.aa": 0,
  }) == {
      "lora00": True,
      "lora1": True,
      "lora1x": False,
      "lora1": True,
      "xx.lora": False,
      "xx.lora3.aa": True,
  }
