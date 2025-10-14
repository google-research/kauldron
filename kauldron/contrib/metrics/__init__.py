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
  from kauldron.contrib.metrics.classification import JointAccuracy
  from kauldron.contrib.metrics.depth_estimation import MeanAbsoluteRelativeError
  from kauldron.contrib.metrics.depth_estimation import RGBMeanAbsoluteRelativeError
  from kauldron.contrib.metrics.depth_estimation import Delta1
  from kauldron.contrib.metrics.point_tracking import TapAverageJaccard
  from kauldron.contrib.metrics.point_tracking import TapOcclusionAccuracy
  from kauldron.contrib.metrics.point_tracking import TapPositionAccuracy
  from kauldron.contrib.metrics.point_tracking_3d import Tap3DPositionAccuracy
  from kauldron.contrib.metrics.point_tracking_3d import Tap3DAverageJaccard
  from kauldron.contrib.metrics.regression import RootMeanSquaredError
  from kauldron.contrib.metrics.segmentation_tracking import SegmentationJaccard

  from kauldron.contrib.metrics.similarity import CosineSimilarity
  from kauldron.contrib.metrics.meancov import MeanCov
  from kauldron.contrib.metrics.image import PsnrWithResize
  from kauldron.contrib.metrics.multilabel_average_precision import MultilabelAveragePrecision
  from kauldron.contrib.metrics.pose import RelativeRotationError
  from kauldron.contrib.metrics.pose import RelativeTranslationError
  from kauldron.contrib.metrics.pose import PoseEndPointError
  from kauldron.contrib.metrics.pose import PoseMeanAverageAccuracy

  from kauldron.contrib.metrics.fid import FidWithStats
