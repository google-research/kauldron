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

"""Kontext is a small self-contained library to manipulate nested trees.

* Extract values

"""

# pylint: disable=g-importing-member,unused-import

from kauldron.kontext.annotate import get_keypaths
from kauldron.kontext.annotate import is_key_annotated
from kauldron.kontext.annotate import Key
from kauldron.kontext.annotate import KeyTree
from kauldron.kontext.annotate import REQUIRED
from kauldron.kontext.annotate import resolve_from_keyed_obj
from kauldron.kontext.annotate import resolve_from_keypaths
from kauldron.kontext.filter_utils import filter_by_path
from kauldron.kontext.glob_paths import GlobPath
from kauldron.kontext.glob_paths import set_by_path
from kauldron.kontext.path_builder import path_builder_from
from kauldron.kontext.paths import Context
from kauldron.kontext.paths import flatten_with_path
from kauldron.kontext.paths import get_by_path
from kauldron.kontext.paths import Path
