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

"""Contrib datasets and pre-processing ops."""

from kauldron.contrib import _lazy_imports

# pylint: disable=g-importing-member,g-import-not-at-top

with _lazy_imports.lazy_api_imports(globals()):

  # Iterators
  from kauldron.contrib.data import iterators as iters

  # Access to contrib PyGrain transforms.
  # Add PyGrain transforms to kd.contrib.data.py.preprocessing
  from kauldron.contrib.data import py

  # Transforms
  # start
  from kauldron.contrib.data.extra_image_ops import RandAugment
  from kauldron.contrib.data.preprocessing import AddConstants
  from kauldron.contrib.data.preprocessing import AddStringField
  from kauldron.contrib.data.preprocessing import ApplyW
  from kauldron.contrib.data.preprocessing import Binarize
  from kauldron.contrib.data.preprocessing import CreateMask
  from kauldron.contrib.data.preprocessing import CreateMeshGridMask
  from kauldron.contrib.data.preprocessing import CreateNonEqualMask
  from kauldron.contrib.data.preprocessing import CreateRangeMask
  from kauldron.contrib.data.preprocessing import ElementWiseRandomTransformWithPredicate
  from kauldron.contrib.data.preprocessing import ElementWiseTransformWithPredicate
  from kauldron.contrib.data.preprocessing import ExtractInitialFixedLengthClip
  from kauldron.contrib.data.preprocessing import FetchElementZero
  from kauldron.contrib.data.preprocessing import FlattenVideo
  from kauldron.contrib.data.preprocessing import FlipUpsideDown
  from kauldron.contrib.data.preprocessing import GaussianBlur
  from kauldron.contrib.data.preprocessing import HStack
  from kauldron.contrib.data.preprocessing import JaxImageResize
  from kauldron.contrib.data.preprocessing import MaskedVideoVisualization
  from kauldron.contrib.data.preprocessing import Normalize
  from kauldron.contrib.data.preprocessing import OneMinus
  from kauldron.contrib.data.preprocessing import PadAxisToFixedSize
  from kauldron.contrib.data.preprocessing import PadImage
  from kauldron.contrib.data.preprocessing import PadImageEdgeVal
  from kauldron.contrib.data.preprocessing import RandomDropTokens
  from kauldron.contrib.data.preprocessing import RandomFlipLeftRightVideo
  from kauldron.contrib.data.preprocessing import RandomResize
  from kauldron.contrib.data.preprocessing import RandomSubsetAlongAxis
  from kauldron.contrib.data.preprocessing import Repeat
  from kauldron.contrib.data.preprocessing import RepeatFrames
  from kauldron.contrib.data.preprocessing import ReshapeSpatialDim
  from kauldron.contrib.data.preprocessing import Resize
  from kauldron.contrib.data.preprocessing import Scale
  from kauldron.contrib.data.preprocessing import SliceVideosIntoFrames
  from kauldron.contrib.data.preprocessing import SliceWithStride
  from kauldron.contrib.data.preprocessing import SpacetimeToDepth
  from kauldron.contrib.data.preprocessing import Standardize
  from kauldron.contrib.data.preprocessing import SubsampleAndFlatten
  from kauldron.contrib.data.preprocessing import TemporalRandomStridedWindow
  from kauldron.contrib.data.preprocessing import TemporalRandomWalk
  from kauldron.contrib.data.preprocessing import TemporalRandomWindow
  from kauldron.contrib.data.preprocessing import TimeChunkedFlattenVideo
  from kauldron.contrib.data.preprocessing import VStack
  from kauldron.contrib.data.preprocessing import ValueRange
  from kauldron.contrib.data.preprocessing import VideoMAENormalization
  from kauldron.contrib.data.scenic_ops import ThreeSpatialCrop
  # end

  # Pose transforms
  from kauldron.contrib.data.pose_preprocessing import TartanPoseToRT
  from kauldron.contrib.data.pose_preprocessing import RelativePoses

  # Frame sampling
  from kauldron.contrib.data.frame_sampling import RandomClip
  from kauldron.contrib.data.frame_sampling import FirstClip
  from kauldron.contrib.data.frame_sampling import MiddleClip
  from kauldron.contrib.data.frame_sampling import MultiClip
