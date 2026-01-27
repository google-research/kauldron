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

"""Tests for the PyTree type annotation."""

from kauldron.ktyping import pytree
import pytest


def test_simple_pytree_type_creation():
  p = pytree.PyTree[int]
  assert isinstance(p, type)
  assert p.leaf_type == int
  assert p.__name__ == "PyTree[int]"
  assert repr(p) == "PyTree[int]"


def test_pytree_creation_with_structure():
  with pytest.raises(
      TypeError, match="structure specification is not supported"
  ):
    _ = pytree.PyTree[float, "T"]


def test_pytree_isinstance_leaftype_only():

  assert isinstance(7, pytree.PyTree[int])
  assert isinstance((1, 2, 3), pytree.PyTree[int])
  assert isinstance({"a": 1, "b": 2}, pytree.PyTree[int])
  assert isinstance({"a": [1], "b": {"c": (2, 3), "d": 4}}, pytree.PyTree[int])

  assert not isinstance(1, pytree.PyTree[str])
  assert not isinstance((1, "2", 3), pytree.PyTree[int])
  assert not isinstance({"a": 1, "b": 2.3}, pytree.PyTree[float])
  assert not isinstance(
      {"a": [1], "b": {"c": (2, 3), "d": "4"}}, pytree.PyTree[int]
  )
