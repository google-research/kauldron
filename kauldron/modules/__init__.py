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

"""Collection of nn.Modules to build neural networks."""

# pylint: disable=g-importing-member,g-bad-import-order

# Do not sort imports
# fmt: skip-import-sorting

from kauldron.modules.pos_embeddings import convert_to_fourier_features
from kauldron.utils.train_property import set_train_property
from kauldron.utils.train_property import train_property
from kauldron.utils.interms_property import interms_property

# module types
from kauldron.modules.knn_types import AttentionModule
from kauldron.modules.knn_types import NormModule
from kauldron.modules.knn_types import ImageTokenizer
from kauldron.modules.knn_types import TransformerBlock

# Modules
from kauldron.modules.adapter import ExternalModule
from kauldron.modules.adapter import WrapperModule
from kauldron.modules.misc import Dropout
from kauldron.modules.misc import DummyModel
from kauldron.modules.misc import Identity
from kauldron.modules.misc import Rearrange
from kauldron.modules.misc import Reduce
from kauldron.modules.pos_embeddings import AddEmbedding
from kauldron.modules.pos_embeddings import AddLearnedEmbedding
from kauldron.modules.pos_embeddings import FourierEmbedding
from kauldron.modules.pos_embeddings import LearnedEmbedding
from kauldron.modules.pos_embeddings import ZeroEmbedding
from kauldron.modules.models import FlatAutoencoder
from kauldron.modules.models import Sequential

# Models should be open-sourced on a individual basis
from kauldron.modules.attention import ImprovedMultiHeadDotProductAttention
from kauldron.modules.attention import MultiHeadDotProductAttention
from kauldron.modules.input_embeddings import Patchify
from kauldron.modules.input_embeddings import PatchifyEmbed

# transformer
from kauldron.modules.transformers import PreNormBlock
from kauldron.modules.transformers import PostNormBlock
from kauldron.modules.transformers import ParallelAttentionBlock
from kauldron.modules.transformers import TransformerMLP
# vit
from kauldron.modules.vit import Vit
from kauldron.modules.vit import VitEncoder
