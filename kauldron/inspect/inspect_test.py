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

from kauldron import inspect
import numpy as np


def test_batch():

  batch = {
      'a': np.asarray(['a', 'bb']),
      'b': np.asarray([1, 2, 3]),
      'c': {
          'f': np.asarray([True, False]),
          'g': np.asarray(1),
      },
  }

  df = inspect.get_batch_stats(batch)
  assert len(df) == 4
  assert list(df['Name']) == ['batch.a', 'batch.b', 'batch.c.f', 'batch.c.g']
