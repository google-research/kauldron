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

"""Test."""

import pickle

import cloudpickle
from etils import epy
import jax
from kauldron.konfig import immutabledict_lib
import pytest


def test_dict():
  d = immutabledict_lib.ImmutableDict({
      'z': 1,
      'a': 2,
      'w': 3,
  })
  d = jax.tree.map(lambda x: x * 10, d)
  assert d == immutabledict_lib.ImmutableDict({
      'z': 10,
      'a': 20,
      'w': 30,
  })
  assert list(d) == ['z', 'a', 'w']  # Key order preserved
  assert d.a == 20  # Attribute access
  assert hash(d)


def test_dict_repr():
  d = immutabledict_lib.ImmutableDict({
      'z': 1,
      'a': 2,
      'w': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  })

  assert repr(d) == epy.dedent("""
      ImmutableDict({
          'z': 1,
          'a': 2,
          'w': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      })
  """)


@pytest.mark.parametrize('pkl', [pickle, cloudpickle])
def test_dict_pickle(pkl):
  a = immutabledict_lib.ImmutableDict({
      'z': 1,
      'a': 2,
  })
  serialized = pkl.dumps(a)
  b = pkl.loads(serialized)
  assert a == b
