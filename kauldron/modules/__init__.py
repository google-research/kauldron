# Copyright 2023 The kauldron Authors.
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

"""Collection of nn.Modules to build neural networks."""

# pylint: disable=g-importing-member,g-bad-import-order

# Do not sort imports
# fmt: skip-import-sorting

from kauldron.klinen.module import Module
from kauldron.utils.train_property import train_property
from kauldron.utils.interms_property import interms_property

# Module Types
from kauldron.modules.knn_types import NormModule
from kauldron.modules.knn_types import PositionEmbedding

# Modules
from kauldron.modules.misc import Dropout
from kauldron.modules.misc import Identity
from kauldron.modules.pos_embeddings import AddEmbedding
from kauldron.modules.pos_embeddings import AddLearnedEmbedding
from kauldron.modules.pos_embeddings import LearnedEmbedding
from kauldron.modules.pos_embeddings import ZeroEmbedding
