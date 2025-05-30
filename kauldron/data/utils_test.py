# Copyright 2025 The kauldron Authors.
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

from etils import enp
from kauldron.data import utils as data_utils
import numpy as np


def test_json_export_import():
  spec = {
      'img': enp.ArraySpec(shape=(64, 64, 3), dtype=np.uint8),
      'label': enp.ArraySpec(shape=(), dtype=np.float32),
  }
  spec_json = data_utils.spec_to_json(spec)
  spec_restored = data_utils.json_to_spec(spec_json)
  assert spec_restored == spec
