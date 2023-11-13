# Copyright 2023 The kauldron Authors.
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

"""Test."""

import dataclasses

from kauldron import kontext
import pytest


@dataclasses.dataclass(frozen=True)
class A:
  x: kontext.Key = kontext.REQUIRED
  y: None | kontext.Key = None


def test_missing():
  tree = {'a': 1, 'b': 2, 'c': 3}

  with pytest.raises(ValueError, match='required keys'):
    kontext.get_from_keys_obj(tree, A())

  with pytest.raises(ValueError, match='required keys'):
    kontext.get_from_keys_obj(tree, A(y='a'))

  assert kontext.get_from_keys_obj(tree, A(x='a')) == {'x': 1, 'y': None}
