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

"""PyGrain public API."""

# pylint: disable=g-importing-member,g-bad-import-order

from kauldron.data.py.base import PyGrainPipeline
from kauldron.data.py.base import DataSourceBase
from kauldron.data.py.data_sources import DataSource
from kauldron.data.py.data_sources import Tfds
from kauldron.data.py.data_sources import Json
from kauldron.data.py.data_sources import HuggingFace
from kauldron.data.py.mixtures import Mix

# *****************************************************************************
# DO NOT ADD new preprocessing ops here. Instead, add them to `kd.contrib.data`
# *****************************************************************************

# ====== Structure transforms ======
from kauldron.data.transforms.base import AddConstants
from kauldron.data.transforms.base import Elements
from kauldron.data.transforms.base import ElementWiseTransform
from kauldron.data.transforms.base import TreeFlattenWithPath
from kauldron.data.py.transform_utils import SliceDataset
# ====== Random transforms ======
# ====== Map transforms ======
from kauldron.data.transforms.map_transforms import Cast
from kauldron.data.transforms.map_transforms import Gather
from kauldron.data.transforms.map_transforms import Rearrange
from kauldron.data.transforms.map_transforms import Resize
from kauldron.data.transforms.map_transforms import ValueRange
