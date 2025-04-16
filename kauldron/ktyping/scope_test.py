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

"""Tests for scope.py."""

import threading
from kauldron.ktyping import errors
from kauldron.ktyping import scope
import pytest


def test_get_current_scope_returns_parent_scope_if_subscope_ok():
  # ---- no active scope -> error  ----
  with pytest.raises(errors.NoActiveScopeError):
    scope.get_current_scope()

  def f(subscope_ok: bool):
    return scope.get_current_scope(subscope_ok=subscope_ok)

  # ---- with active scope  ----
  with scope.ShapeScope() as sscope:
    assert scope.get_current_scope() == sscope

    # nested in function -> error if not subscope_ok
    assert f(subscope_ok=True) == sscope

    with pytest.raises(errors.NoActiveScopeError):
      f(subscope_ok=False)


def test_memo_stack_is_thread_local():
  """Ensures that the memo stack is independent for each thread."""

  def worker_thread():
    # In the worker thread, there should be no active scope
    with pytest.raises(errors.NoActiveScopeError):
      scope.get_current_scope(subscope_ok=True)

    # Open a new ShapeScope and check that it works
    with scope.ShapeScope(alternatives=[{"a": (1,)}]) as worker_sscope:
      assert scope.get_current_scope() is worker_sscope
      assert worker_sscope.dims["a"] == 1

  # In the main thread, start by pushing a different Memo object
  with scope.ShapeScope(alternatives=[{"b": (2,)}]) as main_sscope:
    assert scope.get_current_scope() is main_sscope
    assert main_sscope.dims["b"] == 2
    # Start the worker thread
    worker = threading.Thread(target=worker_thread)
    worker.start()
    assert scope.get_current_scope() is main_sscope
    worker.join()  # Wait for the worker thread to finish
    # After the worker thread has finished, the main thread's memo stack
    # should still contain the Memo object that was pushed in the main thread.
    assert scope.get_current_scope() is main_sscope
    assert main_sscope.dims["b"] == 2


def test_dim_scope_basic_dim_access():
  with scope.ShapeScope() as sscope:
    with pytest.raises(KeyError):
      _ = sscope.dims["a"]
    with pytest.raises(errors.UnknownDimensionError):
      _ = sscope.dims.a

    sscope.dims["a"] = 7
    sscope.dims.b = 9

    assert sscope.dims["a"] == 7
    assert sscope.dims.a == 7
    assert sscope.dims["b"] == 9
    assert sscope.dims.b == 9

    del sscope.dims["a"]

    with pytest.raises(KeyError):
      _ = sscope.dims["a"]

    del sscope.dims.b
    with pytest.raises(KeyError):
      _ = sscope.dims["b"]


def test_dim_scope_multi_dim_access():
  with scope.ShapeScope() as sscope:
    # cannot assign a tuple to a single dim
    with pytest.raises(ValueError):
      sscope.dims["b"] = (8, 9)
    # cannot assign int to a multi-dim
    with pytest.raises(ValueError):
      sscope.dims["*a"] = 5

    sscope.dims["*a"] = (5,)  # can assign single-dim tuple to multi-dim
    sscope.dims["*b"] = (8, 9)
    sscope.dims["c"] = 11

    assert sscope.dims["*b"] == (8, 9)
    assert sscope.dims["*a"] == (5,)

    with pytest.raises(ValueError):
      _ = sscope.dims["b"]  # cannot access a multi-dim as a single dim

    with pytest.raises(ValueError):
      _ = sscope.dims.b  # cannot access a multi-dim as a single dim

    assert (
        sscope.dims["a"] == 5
    )  # can access a length 1 multi-dim as a single dim
    assert sscope.dims.a == 5
    assert sscope.dims["*c"] == (11,)  # can access a single-dim as a multi-dim

    del sscope.dims.a
    del sscope.dims["*b"]
    del sscope.dims["*c"]

    with pytest.raises(KeyError):
      _ = sscope.dims["a"]

    with pytest.raises(KeyError):
      _ = sscope.dims["b"]

    with pytest.raises(KeyError):
      _ = sscope.dims["c"]

    with pytest.raises(KeyError):
      del sscope.dims.non_existent

    with pytest.raises(KeyError):
      del sscope.dims["non_existent"]


def test_dim_scope_access_ambiguous_dim():
  alternative1 = {"a": (1,), "b": (2,)}
  alternative2 = {"a": (3,), "b": (2,), "c": (8,)}
  with scope.ShapeScope(alternatives=[alternative1, alternative2]) as sscope:
    # can access unambiguous dim
    assert sscope.dims["b"] == 2

    # cannot access ambiguous dim
    with pytest.raises(errors.AmbiguousDimensionError):
      _ = sscope.dims["a"]

    # cannot access partially defined dim
    with pytest.raises(errors.AmbiguousDimensionError):
      _ = sscope.dims["c"]


def test_dim_scope_set_ambiguous_dim():
  alternative1 = {"a": (1,), "b": (2,)}
  alternative2 = {"a": (3,), "b": (2,), "c": (8,)}
  with scope.ShapeScope(alternatives=[alternative1, alternative2]) as sscope:
    # cannot set unambiguous dim (no implicit overwriting)
    with pytest.raises(errors.IncompatibleDimensionError):
      sscope.dims.b = 7

    # cannot set partial dim if value is incompatible
    with pytest.raises(errors.IncompatibleDimensionError):
      sscope.dims.c = 17

    # CAN set partial dim if value is compatible
    sscope.dims.c = 8
    assert all(alt["c"] == (8,) for alt in sscope.alternatives)

    # can set novel dim
    sscope.dims.d = 13
    assert all(alt["d"] == (13,) for alt in sscope.alternatives)

    # cannot set ambiguous dim
    with pytest.raises(errors.IncompatibleDimensionError):
      sscope.dims.a = 4


def test_dim_scope_delete_ambiguous_dim():
  alternative1 = {"a": (1,), "b": (2,)}
  alternative2 = {"a": (3,), "b": (2,), "c": (8,)}
  with scope.ShapeScope(alternatives=[alternative1, alternative2]) as sscope:
    # can delete unambiguous dim
    del sscope.dims.b
    assert all("b" not in alt for alt in sscope.alternatives)

    # can delete partial dim
    del sscope.dims.c
    assert all("c" not in alt for alt in sscope.alternatives)

    # can delete ambiguous dim  # TODO is this desired behavior?
    del sscope.dims.a
    assert all("a" not in alt for alt in sscope.alternatives)


def test_can_still_access_attributes():
  with scope.ShapeScope() as sscope:
    sscope.alternatives = [{"a": (1,)}]
    assert all(alt["a"] == (1,) for alt in sscope.alternatives)
