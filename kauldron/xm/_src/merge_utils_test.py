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

"""Test merge jobs."""

import dataclasses
import inspect
from typing import Any

from etils import epy
from kauldron import kxm
from kauldron.xm._src import merge_utils
import pytest
from xmanager import xm
from xmanager import xm_abc


def test_merge():
  @merge_utils.add_merge_support
  class A:

    def __init__(self, a=0, b=1, *, c=2, **others):  # pylint: disable=unused-argument
      self.a = a
      self.b = b
      self.c = c
      self.others = others

  a0 = A('a', new='new', d={'aa': 123})
  a1 = A(d={'bb': 123})
  am = merge_utils.merge(a0, a1)
  assert am.a == 'a'
  assert am.b == 1
  assert am.c == 2
  assert am.others == {
      'new': 'new',
      'd': {'aa': 123, 'bb': 123},
  }


def test_merge_error():

  @merge_utils.add_merge_support
  @dataclasses.dataclass(frozen=True)
  class A:
    w: Any = 0
    x: int = 1
    y: int = 2
    z: dict[str, int] = dataclasses.field(default_factory=dict)

  a0 = A(x=10, z={'a': 30})
  a1 = A(y=20, z={'b': 30})
  assert merge_utils.merge(a0, a1) == A(
      w=0,
      x=10,
      y=20,
      z={
          'a': 30,
          'b': 30,
      },
  )

  a0 = A(w='a')
  a1 = A(w=1)
  with pytest.raises(TypeError, match='different types '):
    merge_utils.merge(a0, a1)

  a0 = A(w=2)
  a1 = A(w=1)
  with pytest.raises(ValueError, match='conflicting values'):
    merge_utils.merge(a0, a1)


def test_extract_passed_kwargs():
  def fn(a, b=1, *, c=2, **others):  # pylint: disable=unused-argument
    pass

  sig = inspect.signature(fn)

  params = sig.bind(1, 2, c=4, d=5, y=6, others=1)
  assert merge_utils._extract_passed_kwargs(sig, params) == {
      'a': 1,
      'b': 2,
      'c': 4,
      'd': 5,
      'y': 6,
      'others': 1,  # kwargs and param match
  }

  params = sig.bind_partial(b=2, y=4)
  assert merge_utils._extract_passed_kwargs(sig, params) == {
      'b': 2,
      'y': 4,
  }


def test_job_params():
  j = kxm.Job()
  with pytest.raises(ValueError, match='dataclasses.replace'):
    dataclasses.replace(j, cell='xx')


def test_repr_only_init():
  assert repr(xm_abc.Borg()) != 'Borg()'
  with merge_utils.repr_only_init():
    assert repr(xm_abc.Borg()) == 'Borg()'


def test_repr():
  # 4 class types
  a0 = xm.JobRequirements(  # Custom class
      replicas=10,
      cpu=5,
      ram=8_000_000,
  )
  a1 = kxm.Debug(dump_hlo=True)  # dataclass
  a2 = kxm.Job(platform='jf=2x2')  # Child of `JobParams`
  a3 = xm_abc.Borg(use_auto_host_resources=True)  # attr class

  # Inside repr_only init
  with merge_utils.repr_only_init():
    assert repr(a0) == epy.dedent("""
    JobRequirements(
        replicas=10,
        cpu=5,
        ram=8000000,
    )
    """)
    assert repr(a1) == 'Debug(dump_hlo=True)'
    assert repr(a2) == "Job(platform='jf=2x2')"
    assert repr(a3) == 'Borg(use_auto_host_resources=True)'

    assert epy.pretty_repr(a0) == epy.dedent("""
    JobRequirements(
        replicas=10,
        cpu=5,
        ram=8000000,
    )
    """)
    assert epy.pretty_repr(a1) == 'Debug(dump_hlo=True)'
    assert epy.pretty_repr(a2) == "Job(platform='jf=2x2')"
    assert epy.pretty_repr(a3) == 'Borg(use_auto_host_resources=True)'

  # Outside repr_only init
  # JobRequirements as a custom `__repr__` already
  assert (
      repr(a0) == 'xm.JobRequirements(cpu=5.0, memory=8000000.0, replicas=10)'
  )
  assert repr(a1) != 'Debug(dump_hlo=True)'
  assert repr(a2) != 'Job(platform="jf=2x2")'
  assert repr(a3) != 'Borg(use_auto_host_resources=True)'

  assert (
      epy.pretty_repr(a0)
      == 'xm.JobRequirements(cpu=5.0, memory=8000000.0, replicas=10)'
  )
  assert epy.pretty_repr(a1).startswith('Debug(\n')
  assert epy.pretty_repr(a2).startswith('Job(\n')
  assert epy.pretty_repr(a3).startswith('Borg(\n')
