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

"""Kauldron `tf.data` public API."""

# pylint: disable=g-importing-member,g-bad-import-order

from kauldron.data.tf.base import TFDataPipeline
from kauldron.data.tf.loaders.graintfds import Tfds
from kauldron.data.tf.loaders.seqio import SeqIOMixture
from kauldron.data.tf.loaders.seqio import SeqIOTask
from kauldron.data.tf.loaders.spec_ds import ElementSpecDataset
from kauldron.data.tf.loaders.tfds_legacy import TfdsLegacy
from kauldron.data.tf.loaders.with_shuffle_buffer import WithShuffleBuffer
from kauldron.data.tf.mixture import SampleFromDatasets
from kauldron.data.tf.mixture import ZipDatasets

# *****************************************************************************
# DO NOT ADD new preprocessing ops here. Instead, add them to `kd.contrib.data`
# *****************************************************************************

# ====== Structure transforms ======
from kauldron.data.transforms.base import AddConstants
from kauldron.data.transforms.base import Elements
from kauldron.data.transforms.base import ElementWiseTransform
from kauldron.data.transforms.base import TreeFlattenWithPath
# ====== Random transforms ======
# TODO(epot): Currently there's no generic `ElementWiseRandomTransform` as
# there's no standardized random API between TF and numpy.
from kauldron.data.tf.random_transforms import ElementWiseRandomTransform
from kauldron.data.tf.random_transforms import InceptionCrop
from kauldron.data.tf.random_transforms import RandomCrop
from kauldron.data.tf.random_transforms import RandomFlipLeftRight
# ====== Map transforms ======
from kauldron.data.transforms.map_transforms import Cast
from kauldron.data.transforms.map_transforms import Gather
from kauldron.data.transforms.map_transforms import Rearrange
from kauldron.data.transforms.map_transforms import ValueRange
from kauldron.data.tf.map_transforms import CenterCrop
from kauldron.data.tf.map_transforms import OneHot
# TODO(epot): Unify Resize & ResizeSmall and have better API. Should replace
# by the generic `kd.data.py.Resize`
from kauldron.data.tf.map_transforms import Resize
from kauldron.data.tf.map_transforms import ResizeSmall
