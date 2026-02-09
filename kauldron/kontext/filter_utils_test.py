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

from kauldron import kontext
import pytest


@pytest.mark.parametrize(
    'glob, input_, expected',
    [
        # Select only the `c` keys.
        (
            '**.c',
            {
                'a': {
                    'b': {
                        'c': 1,
                        'c1': 1,
                        'c2': 1,
                    },
                    'b2': {
                        'c': {'d': 1},
                        'c1': {'d': 1},
                        'c2': {'d': 1},
                    },
                },
                'c': True,
            },
            {
                'a': {
                    'b': {
                        'c': 1,
                    },
                    'b2': {
                        'c': {'d': 1},
                    },
                },
                'c': True,
            },
        ),
        # `**` as leaf
        (
            'a[1].b2.**',
            {
                'a': [
                    {'b2': 1},
                    {
                        'b': {
                            'c': 1,
                        },
                        'b2': [
                            {
                                'c': {'d': 1},
                                'c1': {'d': 1},
                            },
                            4,
                        ],
                    },
                ],
                'b': {'a': 1},
            },
            {
                'a': [
                    {
                        'b2': [
                            {
                                'c': {'d': 1},
                                'c1': {'d': 1},
                            },
                            4,
                        ]
                    }
                ]
            },
        ),
        # `**` singleton
        (
            '**',
            {
                'a': [{'b': 1}, 5],
                'b': [{'b': 1}, 5],
                'c': 1,
            },
            {
                'a': [{'b': 1}, 5],
                'b': [{'b': 1}, 5],
                'c': 1,
            },
        ),
        (
            '**',
            True,
            True,
        ),
        # List reverse selection
        (
            '**.c',
            [
                {},
                {'a': {'c': 1}, 'b': 1},
                {'c': 1},
                {},
                {'a2': {'c': 1, 'b': 1}, 'b': 1},
                {'c': 2},
                {'b': 1},
            ],
            [
                {'a': {'c': 1}},
                {'c': 1},
                {'a2': {'c': 1}},
                {'c': 2},
            ],
        ),
        (
            '**[0]',
            [
                {},
                [0, 1, [2, 3]],
                {},
                {},
                {'a': [1, 2]},
                [[1, 2]],
            ],
            [
                {},
                [0, [2]],
                {'a': [1]},
                # Ambiguity here (both `[[1, 2]]` and `[[1]]` might be
                # expected):
                [[1, 2]],
            ],
        ),
        (
            '**[1]',
            [
                {'a': [0, 1]},
                1,
            ],
            [
                {'a': [1]},
                1,
            ],
        ),
        # Unsupported: Nested `.b.a.b.a`
        # (
        #     '**.b.a',
        #     {
        #         'b': {
        #             'a': 1,
        #             'b': {'a': 2, 'c': 4},
        #             'c': 3,
        #         },
        #     },
        #     {
        #         'b': {
        #             'a': 1,
        #             'b': {'a': 2},
        #         },
        #     },
        # ),
        # Mixing `*` and `**`
        (
            'a.*.b.**',
            {
                'a': [
                    {'b': [1, 2]},
                    {'c': [1, 2], 'b': [1, 2]},
                    {'b': True},
                    {'b': False},
                ],
                'b': {'a': 1},
            },
            {
                'a': [
                    {'b': [1, 2]},
                    {'b': [1, 2]},
                    {'b': True},
                    {'b': False},
                ],
            },
        ),
        # No wildcards
        (
            'a[-1].b',
            {
                'a2': [
                    {'b': 1, 'b2': 2},
                    {'b': 10, 'b2': 2},
                    {'b': 100, 'b2': 2},
                ],
                'a': [
                    {'b': 1, 'b2': 2},
                    {'b': 10, 'b2': 2},
                    {'b': 100, 'b2': 2},
                ],
            },
            {
                'a': [
                    {'b': 100},
                ],
            },
        ),
        # Empty string
        (
            '',
            {
                'a2': [
                    {'b': 1, 'b2': 2},
                ],
                'a': [1, 2],
            },
            {
                'a2': [
                    {'b': 1, 'b2': 2},
                ],
                'a': [1, 2],
            },
        ),
    ],
)
def test_get_glob(glob: str, input_, expected):
  assert kontext.filter_by_path(input_, glob) == expected


def test_get_glob_fail():
  with pytest.raises(KeyError, match='Available keys:'):
    kontext.filter_by_path(
        {
            'a': [
                {'b': [1, 2]},
                {'c': [1, 2]},
                {'b': True},
            ],
        },
        'a.*.b',
    )

  with pytest.raises(KeyError, match='Available keys:'):
    kontext.filter_by_path(
        {
            'a': {
                'b': {
                    'c': 1,
                    'c1': 1,
                },
                'b2': {
                    'c': {'d': 1},
                    'c1': {'d': 1},
                },
            },
            'b': {
                'c1': {'d': 1},
            },
        },
        '**.b.c',
    )
