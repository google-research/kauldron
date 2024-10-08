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

import functools
import flax.struct
from kauldron.metrics import auto_state
from kauldron.metrics import base_state
from kauldron.typing import Float  # pylint: disable=g-multiple-import,member-import
import numpy as np


def test_empty():

  @functools.partial(flax.struct.dataclass, kw_only=True)
  class MyState(auto_state.AutoState):
    a: int = 3
    b: Float = auto_state.sum_field()
    c: Float = auto_state.concat_field()
    d: Float = auto_state.truncate_field(num_field="a")

  empty = MyState.empty()
  assert empty.a is base_state.EMPTY
  assert empty.b is base_state.EMPTY
  assert empty.c is base_state.EMPTY
  assert empty.d is base_state.EMPTY

  s = MyState(a=3, b=np.ones((1, 1)), c=np.ones((2, 2)), d=np.ones((3, 3)))

  s2 = empty.merge(s)
  assert s2.a == 3
  result = s2.compute()
  assert result.b.shape == (1, 1)
  assert result.c.shape == (2, 2)
  assert result.d.shape == (3, 3)

  s3 = s.merge(empty)
  assert s3.a == 3
  result = s3.compute()
  assert result.b.shape == (1, 1)
  assert result.c.shape == (2, 2)
  assert result.d.shape == (3, 3)


def test_merge_sum():
  @functools.partial(flax.struct.dataclass, kw_only=True)
  class SumState(auto_state.AutoState):
    a: int = 3
    b: Float = auto_state.sum_field()

  s1 = SumState(a=3, b=np.ones((3, 2)))
  s2 = SumState(a=3, b=np.ones((3, 2)) * 5)

  s = s1.merge(s2)
  assert s.a == 3
  result = s.compute()
  assert result.b.shape == (3, 2)
  np.testing.assert_allclose(result.b, 6.0)


def test_merge_concat():
  @functools.partial(flax.struct.dataclass, kw_only=True)
  class ConcatState(auto_state.AutoState):
    a: str = "irrelevant"
    b: Float = auto_state.concat_field()
    c: Float = auto_state.concat_field(axis=1)

  s1 = ConcatState(b=np.ones((3, 2)), c=np.ones((3, 2)))
  s2 = ConcatState(b=np.zeros((3, 2)), c=np.zeros((3, 2)))

  s = s1.merge(s2)
  assert s.a == "irrelevant"
  result = s.compute()

  assert result.b.shape == (6, 2)
  assert result.c.shape == (3, 4)
  np.testing.assert_allclose(result.b[:3], 1.0)
  np.testing.assert_allclose(result.b[3:], 0.0)

  np.testing.assert_allclose(result.c[:, :2], 1.0)
  np.testing.assert_allclose(result.c[:, 2:], 0.0)


def test_merge_truncate():
  @functools.partial(flax.struct.dataclass, kw_only=True)
  class TruncateState(auto_state.AutoState):
    num_b: int = 4
    num_c: int = 3
    b: Float = auto_state.truncate_field(num_field="num_b")
    c: Float = auto_state.truncate_field(num_field="num_c", axis=1)

  s1 = TruncateState(b=np.ones((3, 2)), c=np.ones((3, 2)))
  s2 = TruncateState(b=np.ones((3, 2)), c=np.ones((3, 2)))

  s = s1.merge(s2)
  assert s.num_b == 4
  assert s.num_c == 3

  result = s.compute()
  assert result.b.shape == (4, 2)
  assert result.c.shape == (3, 3)
  np.testing.assert_allclose(result.b, 1.0)
  np.testing.assert_allclose(result.c, 1.0)
