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

"""Metrics for clustering tasks."""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

from etils import epy
import flax.struct
from kauldron import kontext
from kauldron.metrics import base
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, Integer, typechecked  # pylint: disable=g-multiple-import,g-importing-member

with epy.lazy_imports():
  from grand_vision.eval.metrics import clustering as gv_clustering  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Ari(base.Metric):
  """Adjusted Rand Index (ARI) computed from predictions and labels.

  ARI is a similarity score to compare two clusterings. ARI returns values in
  the range [-1, 1], where 1 corresponds to two identical clusterings (up to
  permutation), i.e. a perfect match between the predicted clustering and the
  ground-truth clustering. A value of (close to) 0 corresponds to chance.
  Negative values corresponds to cases where the agreement between the
  clusterings is less than expected from a random assignment.

  In this implementation, we use ARI to compare predicted instance segmentation
  masks (including background prediction) with ground-truth segmentation
  annotations.
  """

  num_instances_true: int
  num_instances_pred: int
  ignored_ids: Optional[Sequence[int] | int] = None

  # e.g. "preds.segmentations_video"
  predictions: kontext.Key = kontext.REQUIRED
  labels: kontext.Key = kontext.REQUIRED  # e.g. "batch.segmentations_video"
  mask: Optional[kontext.Key] = None

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      predictions: Integer["*b t h w 1"],
      labels: Integer["*b t h w 1"],
      mask: Optional[Bool["*b 1"] | Float["*b 1"]] = None,
  ) -> Ari.State:
    # TODO(svansteenkiste): support non video inputs.
    # TODO(svansteenkiste): Support video padding mask.
    values = gv_clustering.adjusted_rand_index(
        true_ids=labels[..., 0],
        pred_ids=predictions[..., 0],
        num_instances_true=self.num_instances_true,
        num_instances_pred=self.num_instances_pred,
        ignored_ids=self.ignored_ids,
    )
    return self.State.from_values(values=values, mask=mask)
