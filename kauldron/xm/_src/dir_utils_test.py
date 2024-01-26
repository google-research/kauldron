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

"""Tests."""

from kauldron.xm._src import dir_utils


def test_sweep_kwargs():
  assert (
      dir_utils._format_sweep_kwargs({
          'losses.train[0]': 123,
          'other': [1, 2, 34],
          'test_invalid': 'som//e\n%',
      })
      == 'losses.train(0)=123,other=(1, 2, 34),test_invalid=some'
  )
