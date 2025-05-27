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

"""Test."""

import sys

from kauldron import kd
import pytest


def test_imports():
  assert issubclass(
      kd.contrib.data.TemporalRandomWindow,
      kd.data.tf.ElementWiseRandomTransform,
  )

  # Senic lazyly imported
  assert 'scenic' not in sys.modules
  _ = kd.contrib.data.Dmvr
  assert 'scenic' in sys.modules

  with pytest.raises(
      ImportError, match='Some contrib requires additional deps'
  ):
    _ = kd.contrib.data.AutoDataset
