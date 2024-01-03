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

"""Losses."""

# pylint: disable=g-importing-member
from kauldron.losses.base import compute_losses
from kauldron.losses.base import Loss
from kauldron.losses.simple import L1
from kauldron.losses.simple import L2
from kauldron.losses.simple import MultiClassSigmoidBinaryCrossEntropy
from kauldron.losses.simple import SingleClassSigmoidBinaryCrossEntropy
from kauldron.losses.simple import SoftmaxCrossEntropy
from kauldron.losses.simple import SoftmaxCrossEntropyWithIntLabels
# pylint: enable=g-importing-member
