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

"""Contrib modules."""

from kauldron.contrib import _lazy_imports

# pylint: disable=g-importing-member,g-import-not-at-top

with _lazy_imports.lazy_api_imports(globals()):
  from kauldron.contrib.losses.point import WeightedL1
  from kauldron.contrib.losses.segmentation import DiceLoss
  from kauldron.contrib.losses.segmentation import IoULoss
  from kauldron.contrib.losses.segmentation import SigmoidFocalLoss
