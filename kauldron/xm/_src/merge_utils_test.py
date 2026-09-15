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

"""Test merge jobs."""

import dataclasses
import inspect
from typing import Any

from etils import epy
from kauldron import kxm
from kauldron.xm._src import job_params
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


@dataclasses.dataclass(frozen=True, kw_only=True)
class CustomJob(kxm.Job):
  min_hbm: int = 16
  is_fungible: bool = True


def test_merge_job_subclass_with_params():
  defaults = job_params.JobParams(cell='jn', platform='v4')
  custom_job = CustomJob(target='//path/to:target', min_hbm=32)

  # 1. Merge JobParams defaults with CustomJob
  merged = merge_utils.merge(defaults, custom_job)
  assert isinstance(merged, CustomJob)
  assert merged.cell == 'jn'
  assert merged.platform == 'v4'
  assert merged.target == '//path/to:target'
  assert merged.min_hbm == 32
  assert merged.is_fungible

  # 2. Reverse merge order preserves the subclass
  merged_rev = merge_utils.merge(custom_job, defaults)
  assert isinstance(merged_rev, CustomJob)
  assert merged_rev.cell == 'jn'
  assert merged_rev.platform == 'v4'
  assert merged_rev.target == '//path/to:target'
  assert merged_rev.min_hbm == 32

  # 3. 3-argument merge preserves the concrete subclass across all permutations
  p_cell = job_params.JobParams(cell='jn')
  p_platform = job_params.JobParams(platform='v4')
  for m in (
      merge_utils.merge(custom_job, p_cell, p_platform),
      merge_utils.merge(p_cell, custom_job, p_platform),
      merge_utils.merge(p_cell, p_platform, custom_job),
  ):
    assert isinstance(m, CustomJob)
    assert m.cell == 'jn'
    assert m.platform == 'v4'
    assert m.min_hbm == 32

  # 4. Non-Job parameter containers safely fall back to Job (legacy behavior)
  p0 = job_params.JobParams(cell='jn')
  p1 = job_params.JobParams(platform='v4')
  merged_p = merge_utils.merge(p0, p1)
  assert type(merged_p) is kxm.Job  # pylint: disable=unidiomatic-typecheck
  assert merged_p.cell == 'jn'
  assert merged_p.platform == 'v4'


def test_merge_job_subclass_hierarchy():
  # Base Job with CustomJob preserves the more specific subclass
  base_job = kxm.Job(target='//path/to:target', platform='v4')
  merged_with_base = merge_utils.merge(base_job, CustomJob(min_hbm=64))
  assert isinstance(merged_with_base, CustomJob)
  assert merged_with_base.platform == 'v4'
  assert merged_with_base.min_hbm == 64

  merged_with_base_rev = merge_utils.merge(CustomJob(min_hbm=64), base_job)
  assert isinstance(merged_with_base_rev, CustomJob)
  assert merged_with_base_rev.platform == 'v4'
  assert merged_with_base_rev.min_hbm == 64

  # Multi-level subclass hierarchy
  @dataclasses.dataclass(frozen=True, kw_only=True)
  class SpecificJob(CustomJob):
    extra_tag: str = 'tag'

  spec_job = SpecificJob(min_hbm=64, extra_tag='custom')
  merged_spec = merge_utils.merge(
      CustomJob(target='//path/to:target'), spec_job
  )
  assert isinstance(merged_spec, SpecificJob)
  assert merged_spec.min_hbm == 64
  assert merged_spec.extra_tag == 'custom'
  assert merged_spec.target == '//path/to:target'

  merged_spec_rev = merge_utils.merge(
      spec_job, CustomJob(target='//path/to:target')
  )
  assert isinstance(merged_spec_rev, SpecificJob)
  assert merged_spec_rev.min_hbm == 64
  assert merged_spec_rev.extra_tag == 'custom'
  assert merged_spec_rev.target == '//path/to:target'


def test_merge_job_subclass_with_experiment():
  custom_job = CustomJob(target='//path/to:target', min_hbm=32)
  xp = kxm.Experiment(
      cell='jn',
      platform='v4',
      root_dir='/tmp/test_dir',
      tags=['test'],
  )
  merged_xp = merge_utils.merge(xp, custom_job)
  assert isinstance(merged_xp, CustomJob)
  assert merged_xp.cell == 'jn'
  assert merged_xp.platform == 'v4'
  assert merged_xp.target == '//path/to:target'
  assert merged_xp.min_hbm == 32
  assert not hasattr(merged_xp, 'root_dir')


def test_merge_job_subclass_conflicts():
  # Conflicting field on custom subclass raises ValueError
  with pytest.raises(ValueError, match='conflicting values'):
    merge_utils.merge(CustomJob(min_hbm=16), CustomJob(min_hbm=32))

  # Non-conflicting fields on two custom job instances merge properly
  j1 = CustomJob(min_hbm=32)
  j2 = CustomJob(is_fungible=False)
  merged = merge_utils.merge(j1, j2)
  assert isinstance(merged, CustomJob)
  assert merged.min_hbm == 32
  assert not merged.is_fungible

  # Sibling Job subclasses raise TypeError on conflict
  @dataclasses.dataclass(frozen=True, kw_only=True)
  class SiblingJob(kxm.Job):
    other_field: str = 'val'

  with pytest.raises(
      TypeError,
      match=(
          r'Cannot merge conflicting Job subclasses CustomJob and SiblingJob in'
          r' \(CustomJob / SiblingJob\)'
      ),
  ):
    merge_utils.merge(
        CustomJob(target='//path/to:target', min_hbm=32),
        SiblingJob(target='//path/to:target', other_field='test'),
    )


def test_merge_job_subclass_init_false():
  @dataclasses.dataclass(frozen=True, kw_only=True)
  class JobWithInitFalse(kxm.Job):
    computed: str = dataclasses.field(init=False, default='computed_value')

  @dataclasses.dataclass(frozen=True, kw_only=True)
  class OtherParams(job_params.JobParams):
    computed: str = 'should_be_stripped'

  merged = merge_utils.merge(
      OtherParams(), JobWithInitFalse(target='//path/to:target')
  )
  assert isinstance(merged, JobWithInitFalse)
  assert merged.computed == 'computed_value'
  assert merged.target == '//path/to:target'


def test_merge_job_subclass_missing_dataclass():
  class UndecoratedJob(kxm.Job):
    min_hbm: int = 16

  with pytest.raises(
      TypeError, match='missing the `@dataclasses.dataclass` decorator'
  ):
    merge_utils.merge(UndecoratedJob(), job_params.JobParams(platform='v4'))

  undecorated_field_cls = type(
      'UndecoratedFieldJob',
      (kxm.Job,),
      {'min_hbm': dataclasses.field(default=16)},
  )

  with pytest.raises(
      TypeError, match='missing the `@dataclasses.dataclass` decorator'
  ):
    merge_utils.merge(
        undecorated_field_cls(), job_params.JobParams(platform='v4')
    )
