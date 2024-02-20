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

"""Evaluation metrics for point tracking."""

from __future__ import annotations

import dataclasses

import flax
import jax.numpy as jnp
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Bool, Float, Shape, typechecked  # pylint: disable=g-multiple-import,g-importing-member


def get_evaluation_frames(
    gt_occluded: Float["*B T Q 1"], num_frames: int, query_mode: str
) -> Float["*B T"]:
  """Get which frames to evaluate for point tracking."""
  evaluation_frames = jnp.ones(gt_occluded.shape, dtype=jnp.bool_)
  evaluation_frames.at[:, 0].set(False)  # do not evaluate first frame
  return evaluation_frames


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TapOcclusionAccuracy(metrics.Metric):
  """Occlusion accuracy for point tracking."""

  gt_occluded: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  pred_occluded_logits: kontext.Key = kontext.REQUIRED  # e.g. "pred.visible"
  mask: kontext.Key = kontext.REQUIRED  # e.g. "batch.mask"
  query_mode: str  # e.g. "first" or "strided"

  @flax.struct.dataclass
  class State(metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      gt_occluded: Bool["*B T Q 1"],
      pred_occluded_logits: Float["*B T Q 1"],
      mask: Float["*B T Q 1"],
  ) -> TapOcclusionAccuracy.State:
    gt_occluded = gt_occluded.squeeze(-1).astype(jnp.bool_)
    pred_occluded = pred_occluded_logits.squeeze(-1) > 0
    mask = mask.squeeze(-1)

    gt_visible = jnp.logical_not(gt_occluded)
    pred_visible = jnp.logical_not(pred_occluded)

    evaluation_frames = get_evaluation_frames(
        gt_occluded, Shape("T")[0], self.query_mode
    )

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    values = jnp.equal(pred_visible, gt_visible) & evaluation_frames
    values = jnp.sum(values, axis=-2) * mask[:, 0]
    count = jnp.sum(evaluation_frames, axis=-2) * mask[:, 0]
    values = (values.sum(axis=-1) + 1e-8) / (count.sum(axis=-1) + 1e-8)

    return self.State.from_values(values=values)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TapPositionAccuracy(metrics.Metric):
  """Position accuracy for visible points only."""

  gt_occluded: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  gt_tracks: kontext.Key = kontext.REQUIRED  # e.g. "batch.target_points"
  pred_tracks: kontext.Key = kontext.REQUIRED  # e.g. "pred.tracks"
  mask: kontext.Key = kontext.REQUIRED  # e.g. "batch.mask"
  query_mode: str  # e.g. "first" or "strided"

  @flax.struct.dataclass
  class State(metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      gt_occluded: Bool["*B T Q 1"],
      gt_tracks: Float["*B T Q 2"],
      pred_tracks: Float["*B T Q 2"],
      mask: Float["*B T Q 1"],
  ) -> TapPositionAccuracy.State:
    gt_occluded = gt_occluded.squeeze(-1).astype(jnp.bool_)
    mask = mask.squeeze(-1)

    gt_visible = jnp.logical_not(gt_occluded)

    evaluation_frames = get_evaluation_frames(
        gt_occluded, Shape("T")[0], self.query_mode
    )

    all_frac_within = []
    for thresh in [1, 2, 4, 8, 16]:
      # True positives are points that are within the threshold and where both
      # the prediction and the ground truth are listed as visible.
      within_dist = jnp.sum(
          jnp.square(pred_tracks - gt_tracks), axis=-1
      ) < jnp.square(thresh)
      is_correct = jnp.logical_and(within_dist, gt_visible)

      # Compute the frac_within_threshold, which is the fraction of points
      # within the threshold among points that are visible in the ground truth,
      # ignoring whether they're predicted to be visible.
      count_correct = jnp.sum(is_correct & evaluation_frames, axis=-2)
      count_visible_points = jnp.sum(gt_visible & evaluation_frames, axis=-2)
      count_correct = count_correct * mask[:, 0]
      count_visible_points = count_visible_points * mask[:, 0]
      count_correct = count_correct.sum(axis=-1)
      count_visible_points = count_visible_points.sum(axis=-1)
      frac_correct = (count_correct +  + 1e-8) / (count_visible_points + 1e-8)
      all_frac_within.append(frac_correct)

    values = jnp.mean(jnp.stack(all_frac_within, axis=-1), axis=-1)

    return self.State.from_values(values=values)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TapAverageJaccard(metrics.Metric):
  """Average Jaccard considering both location and occlusion accuracy."""

  gt_occluded: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  gt_tracks: kontext.Key = kontext.REQUIRED  # e.g. "batch.target_points"
  pred_occluded_logits: kontext.Key = kontext.REQUIRED  # e.g. "pred.visible"
  pred_tracks: kontext.Key = kontext.REQUIRED  # e.g. "pred.tracks"
  mask: kontext.Key = kontext.REQUIRED  # e.g. "batch.mask"
  query_mode: str  # e.g. "first" or "strided"

  @flax.struct.dataclass
  class State(metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      gt_occluded: Bool["*B T Q 1"],
      gt_tracks: Float["*B T Q 2"],
      pred_occluded_logits: Float["*B T Q 1"],
      pred_tracks: Float["*B T Q 2"],
      mask: Float["*B T Q 1"],
  ) -> TapAverageJaccard.State:
    gt_occluded = gt_occluded.squeeze(-1).astype(jnp.bool_)
    pred_occluded = pred_occluded_logits.squeeze(-1) > 0
    mask = mask.squeeze(-1)

    gt_visible = jnp.logical_not(gt_occluded)
    pred_visible = jnp.logical_not(pred_occluded)

    evaluation_frames = get_evaluation_frames(
        gt_occluded, Shape("T")[0], self.query_mode
    )

    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
      # True positives are points that are within the threshold and where both
      # the prediction and the ground truth are listed as visible.
      within_dist = jnp.sum(
          jnp.square(pred_tracks - gt_tracks), axis=-1
      ) < jnp.square(thresh)
      is_correct = jnp.logical_and(within_dist, gt_visible)

      true_positives = jnp.sum(
          is_correct & pred_visible & evaluation_frames, axis=-2
      )

      # The denominator of the Jaccard metric is the true positives plus
      # false positives plus false negatives.  However, note that true positives
      # plus false negatives is simply the number of points in the ground truth
      # which is easier to compute than trying to compute all three quantities.
      # Thus we just add the number of points in the ground truth to the number
      # of false positives.
      #
      # False positives are simply points that are predicted to be visible,
      # but the ground truth is not visible or too far from the prediction.
      gt_positives = jnp.sum(gt_visible & evaluation_frames, axis=-2)
      false_positives = (~gt_visible) & pred_visible
      false_positives = false_positives | ((~within_dist) & pred_visible)
      false_positives = jnp.sum(false_positives & evaluation_frames, axis=-2)
      true_positives = true_positives * mask[:, 0]
      gt_positives = gt_positives * mask[:, 0]
      false_positives = false_positives * mask[:, 0]
      true_positives = true_positives.sum(axis=-1)
      gt_positives = gt_positives.sum(axis=-1)
      false_positives = false_positives.sum(axis=-1)
      jaccard = (true_positives + 1e-8) / (
          gt_positives + false_positives + 1e-8
      )
      all_jaccard.append(jaccard)

    values = jnp.mean(jnp.stack(all_jaccard, axis=-1), axis=-1)

    return self.State.from_values(values=values)
