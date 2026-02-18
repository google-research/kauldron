# Copyright 2026 The kauldron Authors.
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

import contextlib
from kauldron.ktyping import frame_utils
import pytest


def test_get_caller_frame():
  this_fn_name = test_get_caller_frame.__name__
  assert frame_utils.get_caller_frame().f_code.co_name == this_fn_name

  # ignores context managers
  with contextlib.nullcontext():
    assert frame_utils.get_caller_frame().f_code.co_name == this_fn_name

  # ignores lambda expressions
  lambda_fn = lambda: frame_utils.get_caller_frame().f_code.co_name
  assert lambda_fn() == this_fn_name

  # ignores list comprehensions
  frame_list = [frame_utils.get_caller_frame().f_code.co_name for _ in range(1)]
  assert frame_list == [this_fn_name]

  # ignores dict comprehensions
  frame_dict = {
      i: frame_utils.get_caller_frame().f_code.co_name for i in range(1)
  }
  assert frame_dict == {0: this_fn_name}

  # ignores generator expressions
  frame_gen = (frame_utils.get_caller_frame().f_code.co_name for _ in range(1))
  assert next(frame_gen) == this_fn_name


def test_get_caller_frame_with_offset():
  def fn0(stacklevel: int):
    return frame_utils.get_caller_frame(stacklevel=stacklevel).f_code.co_name

  def fn1(stacklevel: int):
    with contextlib.nullcontext():  # ignore context manager
      return fn0(stacklevel=stacklevel)

  def fn2(stacklevel: int):
    lambda_fn = lambda: fn1(stacklevel=stacklevel)  # ignore lambda
    return lambda_fn()

  assert fn2(0) == fn0.__name__
  assert fn2(1) == fn1.__name__
  assert fn2(2) == fn2.__name__
  assert fn2(3) == test_get_caller_frame_with_offset.__name__


def test_caller_has_active_scope():
  def fn1(stacklevel: int) -> bool:
    frame = frame_utils.get_caller_frame(stacklevel=stacklevel)
    return frame_utils.has_active_scope(frame)

  def fn2(stacklevel: int) -> bool:
    return fn1(stacklevel=stacklevel)

  def fn3(stacklevel: int) -> bool:
    return fn2(stacklevel=stacklevel)

  # set scope counter to 1 to mark as aktive scope
  __ktyping_scope_counter__ = 1  # pylint: disable=unused-variable

  frame = frame_utils.get_caller_frame()
  assert frame_utils.has_active_scope(frame)

  assert not fn1(0)
  assert fn1(1)
  assert not fn1(2)

  assert not fn2(0)
  assert not fn2(1)
  assert fn2(2)

  assert not fn3(0)
  assert not fn3(1)
  assert not fn3(2)
  assert fn3(3)


def test_assert_caller_has_active_scope():
  # set scope counter to 1 to mark as aktive scope
  __ktyping_scope_counter__ = 1  # pylint: disable=unused-variable

  # no exception because this function has an active scope
  frame_utils.assert_caller_has_active_scope(stacklevel=0)

  def fn(stacklevel: int):
    frame_utils.assert_caller_has_active_scope(stacklevel=stacklevel)

  fn(stacklevel=1)  # also okay

  with pytest.raises(
      frame_utils.NoActiveScopeError,
      match=r">> fn: ---",
  ):
    fn(stacklevel=0)
