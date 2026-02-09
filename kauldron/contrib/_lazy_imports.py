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

"""Lazy imports."""

import functools

from etils import epy

lazy_api_imports = functools.partial(
    epy.lazy_api_imports,
    error_msg=(
        'Failed to import {symbol_name!r}. Some contrib'
        ' requires additional deps'
        '.\n'
    ),
)

lazy_imports = functools.partial(
    epy.lazy_imports,
    error_callback=(
        'Failed to import {symbol_name!r}. Some contrib'
        ' requires additional deps'
        '.\n'
    ),
)
