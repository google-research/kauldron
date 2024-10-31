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


# TODO(epot): Somehow importing kauldron.data.py inside lazy_api_imports create
# infinite recursion when the import is resolved, likely because there's
# some special handling of the suffix `py` to support `third_party.py`. I
# don't have time to investigate so instead the module is imported in
# `lazy_imports` rather than `lazy_api_imports`.
with _epy.lazy_imports():
  # PyGrain based data pipeline.
  from kauldron.data import py

  # tf.data based data pipeline.
  from kauldron.data import tf


with _epy.lazy_api_imports(globals()):
  # Top-level abstractions, independent of any specific backend (TF, PyGrain)
  from kauldron.data.data_utils import IterableDataset
  from kauldron.data.pipelines import Pipeline
  # TODO(epot): Remove `InMemoryPipeline` to use `PyGrain` if PyGrain support
  # efficient batch-lookup.
  from kauldron.data.in_memory import InMemoryPipeline
  from kauldron.data.utils import BatchSize

  # ****************************************************************************
  # DO NOT ADD preprocessing ops here. Instead, add them to `kd.contrib.data`
  # ****************************************************************************

  # User should inherit from those base classes to have transformations
  # supported by both TfGrain (`kd.data.tf`) and PyGrain (`kd.data.py`)
  from kauldron.data.transforms.abc import MapTransform
  # from kauldron.data.transforms.abc import RandomMapTransform
  from kauldron.data.transforms.abc import FilterTransform

  # Transformations can be used in both `kd.data.tf` and `kd.data.py`
  from kauldron.data.transforms.base import Elements
  from kauldron.data.transforms.base import ElementWiseTransform
  from kauldron.data.transforms.base import TreeFlattenWithPath
  from kauldron.data.transforms.map_transforms import Gather
  from kauldron.data.transforms.map_transforms import Rearrange
  from kauldron.data.transforms.map_transforms import ValueRange
