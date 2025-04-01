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

"""One-line documentation for test_tools."""

from typing import Literal
import jax
from kauldron.ktyping import utils
import pytest


class Foo:
  pass


@pytest.mark.parametrize(
    "obj, expected_name",
    [
        (None, "None"),
        (int, "int"),
        (8, "int"),
        ("abc", "str"),
        (Foo, "kauldron.ktyping.utils_test.Foo"),
        (Foo(), "kauldron.ktyping.utils_test.Foo"),
        (list[int], "list[int]"),
        (jax.Array, "jax.Array"),
        (Literal["foo"], "Literal['foo']"),
        (int | None, "int | None"),
    ],
)
def test_get_type_name(obj, expected_name):
  assert utils.get_type_name(obj) == expected_name
