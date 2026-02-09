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

from kauldron import kd
import numpy as np


def test_text():
  metric = kd.summaries.ShowTexts(texts='test')

  state = metric.get_state([
      'Some text',
      'Another text',
      'Text 3',
      'Text 4',
      'Text 5',
      'Text 6',
      'Text 7',
  ])
  state2 = metric.get_state([
      'Text 8',
      'Text 9',
      'Text 10',
      'Text 11',
  ])

  final_state = state.merge(state2)

  out = final_state.compute()

  expected = np.array(
      [
          'Some text',
          'Another text',
          'Text 3',
          'Text 4',
          'Text 5',  # Text truncated to 5
      ],
      dtype=object,
  )
  np.testing.assert_equal(out, expected)
