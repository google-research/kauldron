# Copyright 2023 The kauldron Authors.
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

import pathlib

from etils import epy
from kauldron import konfig


def test_cycles():
  cfg = konfig.ConfigDict({
      'a0': 'abc',
      'c0': {
          'a': 123,
      },
  })
  cfg.a1 = cfg.get_ref('a0')
  cfg.c1 = cfg.get_ref('c0')
  cfg.cycle = cfg

  assert repr(cfg) == epy.dedent("""
    <ConfigDict[&id003 {
        'a0': &id001 'abc',
        'a1': *id001,
        'c0': &id002 {'a': 123},
        'c1': *id002,
        'cycle': *id003,
    }]>
  """)


def test_deserialize():
  cfg = konfig.ConfigDict({
      'paths': [
          {'__const__': 'pathlib:Path'},
          {'__const__': 'pathlib:Path'},
      ],
  })
  cfg = konfig.resolve(cfg)
  assert cfg['paths'] == (pathlib.Path, pathlib.Path)


def test_json_path():
  cfg = konfig.ConfigDict({
      'path': pathlib.Path('a/b'),
  })
  assert cfg.to_json() == '{"path": "a/b"}'
