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

"""Data modules."""

# pylint: disable=g-importing-member
from kauldron.data import deprecated
from kauldron.data.data_utils import IterableDataset
from kauldron.data.grain_utils import MapTransform
from kauldron.data.in_memory import InMemoryPipeline
from kauldron.data.pipelines import Pipeline
from kauldron.data.pipelines import PyGrainPipeline
#
# DO NOT ADD new preprocessing ops here. Instead, add them to `kd.contrib.data`
from kauldron.data.preprocessing import Cast
from kauldron.data.preprocessing import CenterCrop
from kauldron.data.preprocessing import Elements
from kauldron.data.preprocessing import ElementWiseRandomTransform
from kauldron.data.preprocessing import ElementWiseTransform
from kauldron.data.preprocessing import Gather
from kauldron.data.preprocessing import InceptionCrop
from kauldron.data.preprocessing import OneHot
from kauldron.data.preprocessing import RandomCrop
from kauldron.data.preprocessing import RandomFlipLeftRight
from kauldron.data.preprocessing import Rearrange
# TODO(epot): Unify Resize & ResizeSmall and have better API.
from kauldron.data.preprocessing import Resize
from kauldron.data.preprocessing import ResizeSmall
from kauldron.data.preprocessing import TreeFlattenWithPath
from kauldron.data.preprocessing import ValueRange
from kauldron.data.utils import BatchSize
# pylint: enable=g-importing-member
