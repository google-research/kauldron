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

"""Data modules.

"""

# pylint: disable=g-import-not-at-top,g-importing-member,g-bad-import-order

import etils.epy as _epy

with _epy.lazy_api_imports(globals()):
  from kauldron.data.data_utils import IterableDataset
  # Top-level pipelines
  from kauldron.data.pipelines import Pipeline
  from kauldron.data.in_memory import InMemoryPipeline

  # PyGrain based data pipeline.
  from kauldron.data import py

  # User should inherit from those base classes to have transformations
  # supported by both TfGrain (`kd.data.tf`) and PyGrain (`kd.data.py`)
  from kauldron.data.transforms.abc import MapTransform
  # from kauldron.data.transforms.abc import RandomMapTransform
  from kauldron.data.transforms.abc import FilterTransform

  from kauldron.data.utils import BatchSize

  from kauldron.data.py.data_sources import DataSource
  from kauldron.data.py.data_sources import Json
  from kauldron.data.py.data_sources import HuggingFace
  from kauldron.data.py.data_sources import Tfds
  from kauldron.data.py.mixtures import Mix

  # ****************************************************************************
  # DO NOT ADD preprocessing ops here. Instead, add them to `kd.contrib.data`
  # ****************************************************************************

  # Transforms here supports both PyGrain (`kd.data.py`) and `tf.data`
  # (`kd.data.tf`) pipelines.
  # For extra PyGrain-only or TF-only transforms, see `kd.data.py` or
  # `kd.data.tf`

  # ====== Structure transforms ======
  from kauldron.data.transforms.base import AddConstants
  from kauldron.data.transforms.base import Elements
  from kauldron.data.transforms.base import ElementWiseTransform
  from kauldron.data.transforms.base import TreeFlattenWithPath
  # ====== Random transforms ======
  # ====== Map transforms ======
  from kauldron.data.transforms.map_transforms import Cast
  from kauldron.data.transforms.map_transforms import Gather
  from kauldron.data.transforms.map_transforms import Rearrange
  from kauldron.data.transforms.map_transforms import Resize
  from kauldron.data.transforms.map_transforms import ValueRange
