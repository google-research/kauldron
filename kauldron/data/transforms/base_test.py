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

from kauldron import kd
import pytest


def test_elements_keep():
  el = kd.data.py.Elements(
      keep={"yes", "definitely"}, rename={"old": "new"}, copy={"no": "no_copy"}
  )
  before = {"yes": 1, "definitely": 2, "old": 3, "no": 4, "drop": 5}
  after = el.map(before)
  assert set(after.keys()) == {"yes", "definitely", "new", "no_copy"}
  assert after["yes"] == before["yes"]
  assert after["definitely"] == before["definitely"]
  assert after["new"] == before["old"]
  assert after["no_copy"] == before["no"]


def test_elements_drop():
  el = kd.data.py.Elements(
      drop={"no", "drop"}, rename={"old": "new"}, copy={"yes": "yes_copy"}
  )
  before = {"yes": 1, "definitely": 2, "old": 3, "no": 4, "drop": 5}
  after = el.map(before)
  assert set(after.keys()) == {"yes", "definitely", "new", "yes_copy"}
  assert after["yes"] == before["yes"]
  assert after["definitely"] == before["definitely"]
  assert after["new"] == before["old"]
  assert after["yes_copy"] == before["yes"]


def test_elements_rename_only():
  el = kd.data.py.Elements(rename={"old": "new"})
  before = {"yes": 1, "definitely": 2, "old": 3, "no": 4, "drop": 5}
  after = el.map(before)
  assert set(after.keys()) == {"yes", "definitely", "new", "no", "drop"}
  assert after["yes"] == before["yes"]
  assert after["definitely"] == before["definitely"]
  assert after["no"] == before["no"]
  assert after["drop"] == before["drop"]
  assert after["new"] == before["old"]


def test_elements_rename_overwrite_raises():
  el = kd.data.py.Elements(rename={"old": "oops"})
  before = {"old": 1, "oops": 2}
  with pytest.raises(KeyError):
    el.map(before)


def test_elements_copy_only():
  el = kd.data.py.Elements(copy={"yes": "no", "old": "new"})
  before = {"yes": 1, "old": 2}
  after = el.map(before)
  assert set(after.keys()) == {"yes", "no", "old", "new"}
  assert after["yes"] == before["yes"]
  assert after["no"] == before["yes"]
  assert after["old"] == before["old"]
  assert after["new"] == before["old"]


def test_elements_copy_overwrite_raises():
  # copy to an existing key
  el = kd.data.py.Elements(copy={"old": "oops"})
  before = {"old": 1, "oops": 2}
  with pytest.raises(KeyError):
    el.map(before)
  # copy to a key that is also a rename target
  with pytest.raises(KeyError):
    _ = kd.data.py.Elements(copy={"old": "oops"}, rename={"yes": "oops"})
  # copy two fields to the same target name
  with pytest.raises(ValueError):
    _ = kd.data.py.Elements(copy={"old": "oops", "yes": "oops"})
