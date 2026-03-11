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

"""Test."""

from kauldron import konfig
from kauldron import kontext
import pytest


def _assert_path(
    path_str: str,
    ctx,
    expected_ctx=None,
    expected_error=None,
    expected_paths=None,
) -> None:
  path = kontext.GlobPath.from_str(path_str)
  assert str(path) == path_str

  if expected_error:
    with pytest.raises(type(expected_error), match=expected_error.args[0]):
      path.set_in(ctx, 'new')
  else:
    modified = path.set_in(ctx, 'new')
    assert ctx == expected_ctx
    if expected_paths is not None:
      assert sorted(modified) == sorted(expected_paths)


def test_glob_paths():

  _assert_path(
      path_str='a.*[0]',
      ctx={
          'a': {
              'b': [1, 2, 3],
              'b2': [1, 2, 3],
          },
          'b2': 5,
      },
      expected_ctx={
          'a': {
              'b': ['new', 2, 3],
              'b2': ['new', 2, 3],
          },
          'b2': 5,
      },
      expected_paths=['a.b[0]', 'a.b2[0]'],
  )
  _assert_path(
      path_str='**.b',
      ctx={
          'a': {
              'b': [1, 2, 3],
              'b2': [1, 2, 3],
          },
          'b2': 5,
          'b': 'old',
      },
      expected_ctx={
          'a': {
              'b': 'new',
              'b2': [1, 2, 3],
          },
          'b2': 5,
          'b': 'new',
      },
      expected_paths=['b', 'a.b'],
  )
  _assert_path(
      path_str='a.**[0]',
      ctx={
          'a': {
              'b': [1, 2, 3],
              'b2': [2, 2, 3],
              'b3': {
                  'c': [],
                  'c2': [1, 2, 3],
                  'c3': 1,
              },
          },
          'a1': 5,
          'a2': [1, 2, 3],
      },
      expected_ctx={
          'a': {
              'b': ['new', 2, 3],
              'b2': ['new', 2, 3],
              'b3': {
                  'c': [],
                  'c2': ['new', 2, 3],
                  'c3': 1,
              },
          },
          'a1': 5,
          'a2': [1, 2, 3],
      },
      expected_paths=['a.b[0]', 'a.b2[0]', 'a.b3.c2[0]'],
  )
  _assert_path(
      path_str='a.**.b2[0]',
      ctx={
          'a': {
              'b': [1, 2, 3],
              'b2': [2, 2, 3],
              'b3': {
                  'b2': [1, 2, 3],
                  'b3': [1, 2, 3],
                  'c3': 1,
              },
          },
          'a1': {'b2': [1, 2, 3]},
          'a2': [1, 2, 3],
      },
      expected_ctx={
          'a': {
              'b': [1, 2, 3],
              'b2': ['new', 2, 3],
              'b3': {
                  'b2': ['new', 2, 3],
                  'b3': [1, 2, 3],
                  'c3': 1,
              },
          },
          'a1': {'b2': [1, 2, 3]},
          'a2': [1, 2, 3],
      },
      expected_paths=['a.b2[0]', 'a.b3.b2[0]'],
  )


def test_config_dict():

  _assert_path(
      path_str='a.*.b',
      ctx=konfig.ConfigDict({
          'a': [
              konfig.ConfigDict({'b': 'old'}),
              konfig.ConfigDict({'c': 'old'}),
              konfig.ConfigDict({'b': 'old'}),
          ],
      }),
      expected_ctx=konfig.ConfigDict({
          'a': [
              konfig.ConfigDict({'b': 'new'}),
              konfig.ConfigDict({'b': 'new', 'c': 'old'}),
              konfig.ConfigDict({'b': 'new'}),
          ],
      }),
      expected_paths=['a[0].b', 'a[1].b', 'a[2].b'],
  )


def test_glob_end_wildcard():
  """Test."""

  _assert_path(
      path_str='a.*',
      ctx={'a': [1, 2, 3]},
      expected_error=ValueError('Wildcards cannot be located'),
  )
  _assert_path(
      path_str='a.**',
      ctx={'a': [1, 2, 3]},
      expected_error=ValueError('Wildcards cannot be located'),
  )


def test_glob_key_error():
  _assert_path(
      path_str='*.b',
      ctx={
          'a': 1,
          'b': 5,
      },
      expected_error=ValueError('Cannot mutate leaf'),
  )
  _assert_path(
      path_str='*.b',
      ctx={
          'a': {'b': 1},  # All the values should have the `b` property
          'a2': 5,
      },
      expected_error=ValueError('Cannot mutate leaf'),
  )
  _assert_path(
      path_str='*.b',
      ctx={
          'a': {'b': 1},
          'a2': {'b': [2, 3], 'b2': 5},
          'a3': {},
      },
      expected_ctx={
          'a': {'b': 'new'},
          'a2': {'b': 'new', 'b2': 5},
          'a3': {'b': 'new'},
      },
      expected_paths=['a.b', 'a2.b', 'a3.b'],
  )


@pytest.mark.parametrize(
    'glob_str, parent_str',
    [
        ('aa.*.bb.cc', 'aa'),
        ('**.bb.cc', ''),
        ('aa.bb.**.bb.**.cc', 'aa.bb'),
        ('aa[5].*.bb.**.cc', 'aa[5]'),
        ('aa[5].bb.cc', 'aa[5].bb.cc'),
    ],
)
def test_glob_first_non_glob_parent(glob_str, parent_str):
  assert kontext.GlobPath.from_str(
      glob_str
  ).first_non_glob_parent == kontext.Path.from_str(parent_str)


def test_set_by_path_returns_modified_paths():
  ctx = {'a': {'b': [1, 2], 'c': [3, 4]}, 'z': 5}
  modified = kontext.set_by_path(ctx, 'a.*[0]', 'new')
  assert sorted(modified) == ['a.b[0]', 'a.c[0]']

  ctx = {'x': {'y': 'old'}, 'z': 'old2'}
  modified = kontext.set_by_path(ctx, 'x.y', 'new')
  assert modified == ['x.y']

  ctx = {'a': {'b': 'old', 'c': {'b': 'old'}}, 'b': 'old'}
  modified = kontext.set_by_path(ctx, '**.b', 'new')
  assert sorted(modified) == ['a.b', 'a.c.b', 'b']
