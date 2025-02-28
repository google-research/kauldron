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

from kauldron.kontext import path_parser


def test_parse():
  parts = path_parser.parse_parts('a[3][1:2].bbb.*.a')
  assert parts == [
      'a',
      3,
      slice(1, 2),
      'bbb',
      path_parser.Wildcard.STAR,
      'a',
  ]


def test_parse_integer_key():
  parts = path_parser.parse_parts('a.123.b[10]')
  assert parts == [
      'a',
      123,
      'b',
      10,
  ]


def test_parse_interms_example():
  parts = path_parser.parse_parts('interms.model.__call__[0]')
  assert parts == ['interms', 'model', '__call__', 0]
