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

import re
import threading
import kauldron.ktyping as kt
from kauldron.ktyping import errors
from kauldron.ktyping import frame_utils
import pytest


def test_get_current_scope_returns_parent_scope_if_nested_ok():
  # ---- no active scope -> error  ----
  with pytest.raises(frame_utils.NoActiveScopeError):
    kt.get_current_scope()

  def f(nested_ok: bool):
    return kt.get_current_scope(nested_ok=nested_ok)

  # ---- with active scope  ----
  with kt.ShapeScope() as scope:
    assert kt.get_current_scope() == scope

    # nested in function -> error if not nested_ok
    assert f(nested_ok=True) == scope

    with pytest.raises(frame_utils.NoActiveScopeError):
      f(nested_ok=False)


def test_memo_stack_is_thread_local():
  """Ensures that the memo stack is independent for each thread."""

  def worker_thread():
    # In the worker thread, there should be no active scope
    with pytest.raises(frame_utils.NoActiveScopeError):
      kt.get_current_scope(nested_ok=True)

    # Open a new ShapeScope and check that it works
    with kt.ShapeScope(candidates=[{"a": (1,)}]) as worker_scope:
      assert kt.get_current_scope() is worker_scope
      assert worker_scope.dim["a"] == 1

  # In the main thread, start by pushing a different Memo object
  with kt.ShapeScope(candidates=[{"b": (2,)}]) as main_scope:
    assert kt.get_current_scope() is main_scope
    assert main_scope.dim["b"] == 2
    # Start the worker thread
    worker = threading.Thread(target=worker_thread)
    worker.start()
    assert kt.get_current_scope() is main_scope
    worker.join()  # Wait for the worker thread to finish
    # After the worker thread has finished, the main thread's memo stack
    # should still contain the Memo object that was pushed in the main thread.
    assert kt.get_current_scope() is main_scope
    assert main_scope.dim["b"] == 2


def test_dim_scope_basic_dim_access():
  with kt.ShapeScope() as scope:
    with pytest.raises(KeyError, match="Unknown dimension: a"):
      _ = scope.dim["a"]

    scope.dim["a"] = 7

    assert scope.dim["a"] == 7

    del scope.dim["a"]
    with pytest.raises(KeyError):
      _ = scope.dim["a"]


def test_dim_scope_multi_dim_access():
  with kt.ShapeScope() as scope:
    # cannot assign a tuple to a single dim
    with pytest.raises(ValueError):
      scope.dim["b"] = (8, 9)
    # cannot assign int to a multi-dim
    with pytest.raises(ValueError):
      scope.dim["*a"] = 5

    scope.dim["*a"] = (5,)  # can assign single-dim tuple to multi-dim
    scope.dim["*b"] = (8, 9)
    scope.dim["c"] = 11

    assert scope.dim["*b"] == (8, 9)
    assert scope.dim["*a"] == (5,)

    with pytest.raises(ValueError):
      _ = scope.dim["b"]  # cannot access a multi-dim as a single dim

    # can access a length 1 multi-dim as a single dim
    assert scope.dim["a"] == 5
    assert scope.dim["*c"] == (11,)  # can access a single-dim as a multi-dim

    del scope.dim["a"]
    del scope.dim["*b"]
    del scope.dim["*c"]

    with pytest.raises(KeyError):
      _ = scope.dim["a"]

    with pytest.raises(KeyError):
      _ = scope.dim["b"]

    with pytest.raises(KeyError):
      _ = scope.dim["c"]

    with pytest.raises(KeyError):
      del scope.dim["non_existent"]


def test_dim_scope_access_ambiguous_dim():
  candidate1 = {"a": (1,), "b": (2,)}
  candidate2 = {"a": (3,), "b": (2,), "c": (8,)}
  with kt.ShapeScope(candidates=[candidate1, candidate2]) as scope:
    # can access unambiguous dim
    assert scope.dim["b"] == 2

    # cannot access ambiguous dim
    with pytest.raises(errors.AmbiguousDimensionError):
      _ = scope.dim["a"]

    # cannot access partially defined dim
    with pytest.raises(errors.AmbiguousDimensionError):
      _ = scope.dim["c"]


def test_dim_scope_set_ambiguous_dim():
  candidate1 = {"a": (1,), "b": (2,)}
  candidate2 = {"a": (3,), "b": (2,), "c": (8,)}
  with kt.ShapeScope(candidates=[candidate1, candidate2]) as scope:
    # cannot set unambiguous dim (no implicit overwriting)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Incompatible values for 'b' with current_values={(2,)}. Cannot be"
            " assigned to (7,)."
        ),
    ):
      scope.dim["b"] = 7

    # cannot set partial dim if value is incompatible
    with pytest.raises(ValueError, match="Incompatible values for 'c'"):
      scope.dim["c"] = 17

    # CAN set partial dim if value is compatible
    scope.dim["c"] = 8
    assert all(alt["c"] == (8,) for alt in scope.candidates)

    # can set novel dim
    scope.dim["d"] = 13
    assert all(alt["d"] == (13,) for alt in scope.candidates)

    # cannot set ambiguous dim
    with pytest.raises(ValueError, match="Incompatible values for 'a'"):
      scope.dim["a"] = 4


def test_dim_scope_delete_ambiguous_dim():
  candidate1 = {"a": (1,), "b": (2,)}
  candidate2 = {"a": (3,), "b": (2,), "c": (8,)}
  with kt.ShapeScope(candidates=[candidate1, candidate2]) as scope:
    # can delete unambiguous dim
    del scope.dim["b"]
    assert all("b" not in alt for alt in scope.candidates)

    # can delete partial dim
    del scope.dim["c"]
    assert all("c" not in alt for alt in scope.candidates)

    # can delete ambiguous dim
    del scope.dim["a"]
    assert all("a" not in alt for alt in scope.candidates)


def test_can_still_access_attributes():
  with kt.ShapeScope() as scope:
    scope.candidates = [{"a": (1,)}]
    assert all(alt["a"] == (1,) for alt in scope.candidates)


def test_dim():
  with kt.ShapeScope():
    kt.dim["a"] = 1
    assert kt.dim["a"] == 1


def test_dim_contains_with_prefix():
  with kt.ShapeScope(candidates=[{"b": (8, 16)}]) as scope:
    assert "b" in scope.dim
    assert "*b" in scope.dim
    assert "c" not in scope.dim
    assert "*c" not in scope.dim

  with kt.ShapeScope(candidates=[{"a": (5,)}]) as scope:
    assert "a" in scope.dim
    assert "*a" in scope.dim


def test_dim_str_with_partially_defined_candidates():
  candidate1 = {"a": (1,), "b": (2, 3)}
  candidate2 = {"a": (4,)}
  with kt.ShapeScope(candidates=[candidate1, candidate2]) as scope:
    assert "a" in scope.dim
    assert "b" not in scope.dim
