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

"""Ktyping: A library for type annotations of arrays."""

# pylint: disable=g-multiple-import, g-importing-member

from kauldron.ktyping.array_types import *
from kauldron.ktyping.decorator import typechecked
from kauldron.ktyping.dtypes import *
from kauldron.ktyping.scope import ShapeScope, get_current_scope, has_active_scope
from kauldron.ktyping.shape_tools import dim, shape
from kauldron.ktyping.typeguard_checkers import check_type
# pylint: enable=g-multiple-import, g-importing-member
