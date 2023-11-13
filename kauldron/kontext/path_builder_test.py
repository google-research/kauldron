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

"""Tests."""

from __future__ import annotations

from kauldron.kontext import path_builder
import pytest


class A(path_builder.AnnotatedPathBuilder):
  a0: int
  a1: B


class B(path_builder.AnnotatedPathBuilder):
  b0: A


def test_path_builder_annotated():
  a = A()
  assert str(a) == 'A'
  assert str(a.a0) == 'A.a0'
  assert str(a.a1.b0) == 'A.a1.b0'

  with pytest.raises(AttributeError):
    _ = a.non_existing  # pytype: disable=attribute-error

  b = a.a1
  with pytest.raises(AttributeError):
    _ = b.a1  # pytype: disable=attribute-error


def test_dynamic_path_builder():
  x = path_builder.DynamicPathBuilder()
  assert (
      str(x.bbb[123].ccc['aa']['bb'].eee)
      == "DynamicPathBuilder.bbb[123].ccc['aa']['bb'].eee"
  )
