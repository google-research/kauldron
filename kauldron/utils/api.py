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

"""Utils public API."""

# Use `api.py` rather than `__init__.py` to allow importing submodules here
# without triggering a full import.

# pylint: disable=unused-import,g-importing-member,g-import-not-at-top

from etils import epy as _epy

# Namespaces
with _epy.lazy_api_imports(globals()):
  from kauldron.utils import chrono_utils as chrono
  from kauldron.utils import colab
  from kauldron.utils.status_utils import status
