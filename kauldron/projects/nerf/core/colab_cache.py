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

"""Colab cache util."""

import collections
from typing import Any

import __main__  # pylint: disable=g-bad-import-order


def get_cache(obj: Any) -> dict[str, Any]:
  if not hasattr(__main__, '_colab_cache'):
    __main__._colab_cache = collections.defaultdict(dict)  # pylint: disable=protected-access
  return __main__._colab_cache[hash(obj)]  # pylint: disable=protected-access
