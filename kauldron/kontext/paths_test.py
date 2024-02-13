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

"""Property based testing for string paths."""

from typing import Any

import flax.struct
import hypothesis
import hypothesis.strategies as st
from kauldron import kontext
import pytest
import regex


IDENTIFIER_REGEX = regex.compile(r"[^\d\W]\w*")
identifiers = st.from_regex(IDENTIFIER_REGEX, fullmatch=True)
strings = st.text(
    alphabet=st.characters(whitelist_categories=["Lu", "Ll", "Lt", "Lo", "Nl"]),
    max_size=64,
    min_size=0,
)
ints = st.integers()
floats = st.floats(allow_nan=False, allow_infinity=False)
complex_numbers = st.complex_numbers(allow_infinity=False, allow_nan=False)
bools = st.booleans()
nones = st.none()
slices = st.slices(size=11)
simple_literals = (
    strings | ints | floats | complex_numbers | bools | nones | slices
)


@st.composite
def random_length_tuples(
    draw, elements=simple_literals, min_length=0, max_length=3
):
  tuple_len = draw(st.integers(min_value=min_length, max_value=max_length))
  tuple_spec = [elements] * tuple_len
  return draw(st.tuples(*tuple_spec))


literals = st.recursive(simple_literals, random_length_tuples, max_leaves=6)


@hypothesis.given(identifiers)
def test_python_identifier_regex(x: str):
  assert x.isidentifier()


@hypothesis.settings(deadline=None)
@hypothesis.given(st.lists(simple_literals, min_size=1, max_size=16))
def test_path_parsing(path_elements):
  p = kontext.Path(*path_elements)
  p2 = kontext.Path.from_str(repr(p))
  assert p == p2


def test_path_parsing_custom_example():
  path_str = "iterm.encoder.layer_0.MHDP.attention[-1].foo"
  assert str(kontext.Path.from_str(path_str)) == path_str


def test_path_parsing_tensor_slice():
  path_str = "iterm.some.tensor[...,:,:1,7:9,::-1,None]"
  assert str(kontext.Path.from_str(path_str)) == path_str


@pytest.mark.parametrize(
    "path_str",
    [
        "tensor[1 ][ True ][ 10 : 2 : -1  ]",
        "tensor[  :   ,    7  , ::-1 , None ]",
    ],
)
def test_path_parsing_whitespace(path_str):
  path_without_ws = path_str.replace(" ", "")
  assert str(kontext.Path.from_str(path_str)) == path_without_ws


def test_tree_flatten_with_path():
  @flax.struct.dataclass
  class MyTree:
    foo: Any
    bar: Any

  mt = MyTree(foo={"a": 10, "b": [7]}, bar=9)
  flat_tree = kontext.flatten_with_path(mt)
  assert flat_tree == {"foo.a": 10, "foo.b[0]": 7, "bar": 9}

  flat_tree = kontext.flatten_with_path(mt, prefix="cfg")
  assert flat_tree == {"cfg.foo.a": 10, "cfg.foo.b[0]": 7, "cfg.bar": 9}

  flat_tree = kontext.flatten_with_path(mt, separator="/")
  assert flat_tree == {"foo/a": 10, "foo/b/0": 7, "bar": 9}

  flat_tree = kontext.flatten_with_path(mt, prefix="cfg", separator="/")
  assert flat_tree == {"cfg/foo/a": 10, "cfg/foo/b/0": 7, "cfg/bar": 9}
