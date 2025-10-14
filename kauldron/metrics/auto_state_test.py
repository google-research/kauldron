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

import flax.struct
import jax.numpy as jnp
from kauldron.metrics import auto_state
from kauldron.metrics import base_state
from kauldron.typing import Float  # pylint: disable=g-importing-member
import numpy as np
import pytest


@flax.struct.dataclass(kw_only=True)
class _StateFieldInnerState(auto_state.AutoState):
  x: Float = auto_state.sum_field()

  def compute(self):
    return self.x * 2


@flax.struct.dataclass(kw_only=True)
class _StateFieldOuterState(auto_state.AutoState):
  inner: _StateFieldInnerState = auto_state.state_field()
  y: Float = auto_state.sum_field()


def test_empty():

  @flax.struct.dataclass(kw_only=True)
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

  @flax.struct.dataclass(kw_only=True)
  class SumState(auto_state.AutoState):
    a: int = 3
    b: Float = auto_state.sum_field()
    c: Float | None = auto_state.sum_field(default=None)

  s1 = SumState(a=3, b=np.ones((3, 2)))
  s2 = SumState(a=3, b=np.ones((3, 2)) * 5)
  s3 = SumState(a=3, b=np.ones((3, 2)) * 10, c=np.ones((1, 1)))

  s = s1.merge(s2)
  assert s.a == 3
  result = s.compute()
  assert result.b.shape == (3, 2)
  np.testing.assert_allclose(result.b, 6.0)

  # no error:
  _ = s3.merge(s3)

  with pytest.raises(ValueError, match="Cannot sum None"):
    s1.merge(s3)

  with pytest.raises(ValueError, match="Cannot sum None"):
    s3.merge(s1)


def test_merge_concat():

  @flax.struct.dataclass(kw_only=True)
  class ConcatState(auto_state.AutoState):
    a: str = "irrelevant"
    b: Float = auto_state.concat_field()
    c: Float = auto_state.concat_field(axis=1)
    d: Float | None = auto_state.concat_field(default=None)

  s1 = ConcatState(b=np.ones((3, 2)), c=np.ones((3, 2)))
  s2 = ConcatState(b=np.zeros((3, 2)), c=np.zeros((3, 2)))
  s3 = ConcatState(b=np.ones((3, 2)), c=np.ones((3, 2)), d=np.ones((1, 1)))

  s = s1.merge(s2)
  assert s.a == "irrelevant"
  result = s.compute()

  assert result.b.shape == (6, 2)
  assert result.c.shape == (3, 4)
  np.testing.assert_allclose(result.b[:3], 1.0)
  np.testing.assert_allclose(result.b[3:], 0.0)

  np.testing.assert_allclose(result.c[:, :2], 1.0)
  np.testing.assert_allclose(result.c[:, 2:], 0.0)

  # no error:
  _ = s3.merge(s3)

  with pytest.raises(ValueError, match="Cannot concatenate None"):
    s1.merge(s3)

  with pytest.raises(ValueError, match="Cannot concatenate None"):
    s3.merge(s1)


def test_merge_truncate():

  @flax.struct.dataclass(kw_only=True)
  class TruncateState(auto_state.AutoState):
    num_b: int = 4
    num_c: int = 3
    b: Float = auto_state.truncate_field(num_field="num_b")
    c: Float = auto_state.truncate_field(num_field="num_c", axis=1)
    d: Float | None = auto_state.truncate_field(num_field="num_b", default=None)

  s1 = TruncateState(b=np.ones((3, 2)), c=np.ones((3, 2)))
  s2 = TruncateState(b=np.ones((3, 2)), c=np.ones((3, 2)))
  s3 = TruncateState(b=np.ones((3, 2)), c=np.ones((3, 2)), d=np.ones((1, 1)))

  s = s1.merge(s2)
  assert s.num_b == 4
  assert s.num_c == 3

  result = s.compute()
  assert result.b.shape == (4, 2)
  assert result.c.shape == (3, 3)
  np.testing.assert_allclose(result.b, 1.0)
  np.testing.assert_allclose(result.c, 1.0)

  # no error:
  _ = s3.merge(s3)

  with pytest.raises(ValueError, match=r"Cannot .*truncate.* None"):
    s1.merge(s3)

  with pytest.raises(ValueError, match=r"Cannot .*truncate.* None"):
    s3.merge(s1)


def test_merge_truncate_without_merge():

  @flax.struct.dataclass(kw_only=True)
  class TruncateState(auto_state.AutoState):
    num: int = 4
    arr: Float = auto_state.truncate_field(num_field="num")

  s = TruncateState(arr=np.ones((8, 2)))

  result = s.compute()
  # make sure the field is truncated even if it is not merged
  assert result.arr.shape == (4, 2)


def test_merge_sum_tree():

  @flax.struct.dataclass(kw_only=True)
  class SumState(auto_state.AutoState):
    my_tree: dict[str, Float] = auto_state.sum_field()

  s1 = SumState(my_tree={"a": np.ones((3, 2)), "b": np.ones((5,)) * 5})
  s2 = SumState(my_tree={"a": np.ones((3, 2)) * 3, "b": np.ones((5,))})

  s = s1.merge(s2)
  result = s.compute()
  assert result.my_tree["a"].shape == (3, 2)
  assert result.my_tree["b"].shape == (5,)
  np.testing.assert_allclose(result.my_tree["a"], 4.0)
  np.testing.assert_allclose(result.my_tree["b"], 6.0)


def test_merge_min():

  @flax.struct.dataclass(kw_only=True)
  class MinState(auto_state.AutoState):
    b: Float = auto_state.min_field()

  s1 = MinState(b=np.ones((3, 2)))
  s2 = MinState(b=np.ones((3, 2)) * 5)

  s = s1.merge(s2)
  result = s.compute()
  assert result.b.shape == (3, 2)
  np.testing.assert_allclose(result.b, 1.0)


def test_merge_max():

  @flax.struct.dataclass(kw_only=True)
  class MaxState(auto_state.AutoState):
    b: Float = auto_state.max_field()

  s1 = MaxState(b=np.ones((3, 2)))
  s2 = MaxState(b=np.ones((3, 2)) * 5)

  s = s1.merge(s2)
  result = s.compute()
  assert result.b.shape == (3, 2)
  np.testing.assert_allclose(result.b, 5.0)


def test_state_field():
  s1 = _StateFieldOuterState(
      inner=_StateFieldInnerState(x=jnp.array(1.0)), y=jnp.array(2.0)
  )
  s2 = _StateFieldOuterState(
      inner=_StateFieldInnerState(x=jnp.array(10.0)), y=jnp.array(20.0)
  )

  s_merged = s1.merge(s2)
  assert isinstance(s_merged.inner, _StateFieldInnerState)
  np.testing.assert_allclose(s_merged.inner.x, 11.0)
  np.testing.assert_allclose(s_merged.y, 22.0)

  result = s_merged.compute()
  assert result.inner == 22.0  # inner.compute() should be called
  assert result.y == 22.0


def test_finalize():
  @flax.struct.dataclass(kw_only=True)
  class MixedState(auto_state.AutoState):
    n: int = 3
    c: Float = auto_state.concat_field()
    t: Float = auto_state.truncate_field(num_field="n")
    s: Float = auto_state.sum_field()

  s1 = MixedState(c=jnp.ones((2, 3)), t=jnp.ones((2, 5)), s=jnp.ones((2, 7)))
  s2 = MixedState(c=jnp.ones((2, 3)), t=jnp.ones((2, 5)), s=jnp.ones((2, 7)))

  # finalize does not change the static values and the shapes of the arrays
  # for a single (non-merged) state
  s1_final = s1.finalize()
  assert s1_final.n == 3
  assert s1_final.c.shape == (2, 3)
  assert s1_final.t.shape == (2, 5)
  assert s1_final.s.shape == (2, 7)
  # finalize converts jax.Arrays to np.ndarrays
  assert isinstance(s1_final.c, np.ndarray)
  assert isinstance(s1_final.t, np.ndarray)
  assert isinstance(s1_final.s, np.ndarray)

  # finalize should turn tuples/lists into np.ndarrays again
  s = s1.merge(s2)
  s_final = s.finalize()
  assert s_final.n == 3
  assert s_final.c.shape == (4, 3)
  assert s_final.t.shape == (3, 5)
  assert s_final.s.shape == (2, 7)
