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

"""Testing utilities."""

# pylint: disable=g-import-not-at-top,g-importing-member

import sys

pytest = sys.modules.get('pytest')
if pytest:
  # Inside tests, rewrite `assert` statement for better debug messages
  pytest.register_assert_rewrite('kauldron.testing.assert_utils')
  pytest.register_assert_rewrite('kauldron.testing.assert_spec')
del pytest, sys

from kauldron.testing.assert_spec import assert_step_specs
