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

"""Non-default image preprocessing ops (with extra dependencies)."""

from __future__ import annotations

import dataclasses
from typing import Optional

import grain.tensorflow as grain
from kauldron import kontext
from kauldron.typing import TfFloat, TfUInt8, check_type  # pylint: disable=g-multiple-import,g-importing-member
import tensorflow_models as tfm


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandAugment(grain.RandomMapTransform):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719.
  See here for details:
  https://github.com/tensorflow/models/tree/HEAD/tensorflow_models/official/vision/ops/augment.py;l=2331

  Attributes:
    num_layers: The number of augmentation transformations to apply sequentially
      to an image. Represented as (N) in the paper. Usually best values will be
      in the range [1, 3].
    magnitude: Shared magnitude across all augmentation operations. Represented
      as (M) in the paper. Usually best values are in the range [5, 10].
    cutout_const: multiplier for applying cutout.
    translate_const: multiplier for applying translation.
    magnitude_std: randomness of the severity as proposed by the authors of the
      timm library.
    prob_to_apply: The probability to apply the selected augmentation at each
      layer.
    exclude_ops: exclude selected operations.
  """

  image_key: kontext.Key = kontext.REQUIRED  # e.g. "image"
  boxes_key: Optional[kontext.Key] = None

  num_layers: int = 2
  magnitude: int = 10
  cutout_const: float = 40.0
  translate_const: float = 100.0
  magnitude_std: float = 0.0
  prob_to_apply: Optional[float] = None
  exclude_ops: Optional[list[str]] = None

  def random_map(self, features, seed):
    del seed  # TODO(klausg) stateless/deterministic version of this op?
    image = features[self.image_key]
    check_type(image, TfUInt8["h w 3"])

    if self.boxes_key is None:
      randaug = tfm.vision.augment.RandAugment(
          num_layers=self.num_layers,
          magnitude=self.magnitude,
          cutout_const=self.cutout_const,
          translate_const=self.translate_const,
          magnitude_std=self.magnitude_std,
          prob_to_apply=self.prob_to_apply,
          exclude_ops=self.exclude_ops,
      )
      features[self.image_key] = randaug.distort(image)
    else:
      boxes = features[self.boxes_key]
      check_type(boxes, TfFloat["n 4"])
      randaug = tfm.vision.augment.RandAugment.build_for_detection(
          num_layers=self.num_layers,
          magnitude=self.magnitude,
          cutout_const=self.cutout_const,
          translate_const=self.translate_const,
          magnitude_std=self.magnitude_std,
          prob_to_apply=self.prob_to_apply,
          exclude_ops=self.exclude_ops,
      )
      image_aug, boxes_aug = randaug.distort_with_boxes(image, boxes)
      features[self.image_key], features[self.boxes_key] = image_aug, boxes_aug

    return features
