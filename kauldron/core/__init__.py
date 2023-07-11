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

"""Core classes and abstractions used in Kauldron."""
# pylint: disable=g-importing-member
from kauldron.core.annotate import is_key_annotated
from kauldron.core.annotate import resolve_kwargs
from kauldron.core.context import Context
from kauldron.core.paths import get_by_path
from kauldron.core.paths import Path
from kauldron.core.paths import tree_flatten_with_path
# pylint: disable=g-importing-member
