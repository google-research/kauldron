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

"""Evaluation metrics for point tracking. TODO(yiya) move to contrib."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Literal, Optional

import einops
import flax
import jax.numpy as jnp
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Float, Shape, typechecked  # pylint: disable=g-multiple-import,g-importing-member


def get_evaluation_frames(
    query_frame: Float["*B Q"], num_frames: int, query_mode: str
) -> Float["*B Q T"]:
  """Get which frames to evaluate for point tracking."""
  eye = jnp.eye(num_frames, dtype=jnp.int32)
  if query_mode == "first":
    # evaluate frames after the query frame
    query_frame_to_eval_frames = jnp.cumsum(eye, axis=1) - eye
  elif query_mode == "strided":
    # evaluate all frames except the query frame
    query_frame_to_eval_frames = 1 - eye
  else:
    raise ValueError("Unknown query mode " + query_mode)

  query_frame = jnp.round(query_frame).astype(jnp.int32)
  evaluation_frames = query_frame_to_eval_frames[query_frame] > 0
  return evaluation_frames


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TapOcclusionAccuracy(metrics.Metric):
  """Occlusion accuracy for point tracking."""

  pred_visible: kontext.Key = kontext.REQUIRED  # e.g. "pred.visible"
  gt_visible: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  query_frame: Optional[kontext.Key] = None  # e.g. "batch.query_frame"
  query_mask: Optional[kontext.Key] = None  # e.g. "batch.query_mask"
  query_mode: str  # e.g. "first" or "strided"
  axis_order: Literal["BQTC", "BTQC"] = "BQTC"  # TODO(yiya) remove this
  pred_is_logits: bool = False  # TODO(yiya) remove this

  @flax.struct.dataclass
  class State(metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred_visible: Float["*B M N 1"],
      gt_visible: Float["*B M N 1"],
      query_frame: Optional[Float["*B Q"]] = None,
      query_mask: Optional[Float["*B Q"]] = None,
  ) -> TapOcclusionAccuracy.State:
    if self.pred_is_logits:
      pred_visible = pred_visible.squeeze(-1) > 0
    else:
      pred_visible = pred_visible.squeeze(-1).astype(jnp.bool_)
    gt_visible = gt_visible.squeeze(-1).astype(jnp.bool_)

    if query_frame is None:  # Assume query from first frame if not provided
      query_frame = jnp.zeros(Shape("*B Q"))

    if self.axis_order == "BTQC":
      pred_visible = einops.rearrange(pred_visible, "... M N -> ... N M")
      gt_visible = einops.rearrange(gt_visible, "... M N -> ... N M")
      evaluation_frames = get_evaluation_frames(
          query_frame, Shape("M")[0], self.query_mode
      )
    else:
      evaluation_frames = get_evaluation_frames(
          query_frame, Shape("N")[0], self.query_mode)

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    values = jnp.equal(pred_visible, gt_visible) & evaluation_frames
    values = values.sum(axis=-1)
    count = evaluation_frames.sum(axis=-1)
    if query_mask is not None:
      values = values * query_mask
      count = count * query_mask
    values = values.sum(axis=-1) / (count.sum(axis=-1) + 1e-8)

    return self.State.from_values(values=values)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TapPositionAccuracy(metrics.Metric):
  """Position accuracy for visible points only."""

  pred_tracks: kontext.Key = kontext.REQUIRED  # e.g. "pred.tracks"
  gt_tracks: kontext.Key = kontext.REQUIRED  # e.g. "batch.target_points"
  gt_visible: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  query_frame: Optional[kontext.Key] = None  # e.g. "batch.query_frame"
  query_mask: Optional[kontext.Key] = None  # e.g. "batch.query_mask"
  query_mode: str  # e.g. "first" or "strided"
  axis_order: Literal["BQTC", "BTQC"] = "BQTC"  # TODO(yiya) remove this
  # pixel radius to compute accuracy
  thresholds: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0)

  @flax.struct.dataclass
  class State(metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred_tracks: Float["*B M N 2"],
      gt_tracks: Float["*B M N 2"],
      gt_visible: Float["*B M N 1"],
      query_frame: Optional[Float["*B Q"]] = None,
      query_mask: Optional[Float["*B Q"]] = None,
  ) -> TapPositionAccuracy.State:
    gt_visible = gt_visible.squeeze(-1).astype(jnp.bool_)

    if query_frame is None:  # Assume query from first frame if not provided
      query_frame = jnp.zeros(Shape("*B Q"))

    if self.axis_order == "BTQC":
      pred_tracks = einops.rearrange(pred_tracks, "... M N C -> ... N M C")
      gt_tracks = einops.rearrange(gt_tracks, "... M N C -> ... N M C")
      gt_visible = einops.rearrange(gt_visible, "... M N -> ... N M")
      evaluation_frames = get_evaluation_frames(
          query_frame, Shape("M")[0], self.query_mode
      )
    else:
      evaluation_frames = get_evaluation_frames(
          query_frame, Shape("N")[0], self.query_mode
      )

    all_frac_within = []
    for thresh in self.thresholds:
      # True positives are points that are within the threshold and where both
      # the prediction and the ground truth are listed as visible.
      within_dist = jnp.sum(
          jnp.square(pred_tracks - gt_tracks), axis=-1
      ) < jnp.square(thresh)
      is_correct = jnp.logical_and(within_dist, gt_visible)

      # Compute the frac_within_threshold, which is the fraction of points
      # within the threshold among points that are visible in the ground truth,
      # ignoring whether they're predicted to be visible.
      count_correct = jnp.sum(is_correct & evaluation_frames, axis=-1)
      count_visible_points = jnp.sum(gt_visible & evaluation_frames, axis=-1)
      if query_mask is not None:
        count_correct = count_correct * query_mask
        count_visible_points = count_visible_points * query_mask
      count_correct = count_correct.sum(axis=-1)
      count_visible_points = count_visible_points.sum(axis=-1)
      frac_correct = count_correct / (count_visible_points + 1e-8)
      all_frac_within.append(frac_correct)

    values = jnp.mean(jnp.stack(all_frac_within, axis=-1), axis=-1)

    return self.State.from_values(values=values)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TapAverageJaccard(metrics.Metric):
  """Average Jaccard considering both location and occlusion accuracy."""

  pred_tracks: kontext.Key = kontext.REQUIRED  # e.g. "pred.tracks"
  pred_visible: kontext.Key = kontext.REQUIRED  # e.g. "pred.visible"
  gt_tracks: kontext.Key = kontext.REQUIRED  # e.g. "batch.target_points"
  gt_visible: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  query_frame: Optional[kontext.Key] = None  # e.g. "batch.query_frame"
  query_mask: Optional[kontext.Key] = None  # e.g. "batch.query_mask"
  query_mode: str  # e.g. "first" or "strided"
  axis_order: Literal["BQTC", "BTQC"] = "BQTC"  # TODO(yiya) remove this
  pred_is_logits: bool = False  # TODO(yiya) remove this
  # pixel radius to compute accuracy
  thresholds: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0)

  @flax.struct.dataclass
  class State(metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred_tracks: Float["*B M N 2"],
      pred_visible: Float["*B M N 1"],
      gt_tracks: Float["*B M N 2"],
      gt_visible: Float["*B M N 1"],
      query_frame: Optional[Float["*B Q"]] = None,
      query_mask: Optional[Float["*B Q"]] = None,
  ) -> TapAverageJaccard.State:
    if self.pred_is_logits:
      pred_visible = pred_visible.squeeze(-1) > 0
    else:
      pred_visible = pred_visible.squeeze(-1).astype(jnp.bool_)
    gt_visible = gt_visible.squeeze(-1).astype(jnp.bool_)

    if query_frame is None:  # Assume query from first frame if not provided
      query_frame = jnp.zeros(Shape("*B Q"))

    if self.axis_order == "BTQC":
      pred_tracks = einops.rearrange(pred_tracks, "... M N C -> ... N M C")
      pred_visible = einops.rearrange(pred_visible, "... M N -> ... N M")
      gt_tracks = einops.rearrange(gt_tracks, "... M N C -> ... N M C")
      gt_visible = einops.rearrange(gt_visible, "... M N -> ... N M")
      evaluation_frames = get_evaluation_frames(
          query_frame, Shape("M")[0], self.query_mode
      )
    else:
      evaluation_frames = get_evaluation_frames(
          query_frame, Shape("N")[0], self.query_mode
      )

    all_jaccard = []
    for thresh in self.thresholds:
      # True positives are points that are within the threshold and where both
      # the prediction and the ground truth are listed as visible.
      within_dist = jnp.sum(
          jnp.square(pred_tracks - gt_tracks), axis=-1
      ) < jnp.square(thresh)
      is_correct = jnp.logical_and(within_dist, gt_visible)

      true_positives = jnp.sum(
          is_correct & pred_visible & evaluation_frames, axis=-1
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
      gt_positives = jnp.sum(gt_visible & evaluation_frames, axis=-1)
      false_positives = (~gt_visible) & pred_visible
      false_positives = false_positives | ((~within_dist) & pred_visible)
      false_positives = jnp.sum(false_positives & evaluation_frames, axis=-1)
      if query_mask is not None:
        true_positives = true_positives * query_mask
        gt_positives = gt_positives * query_mask
        false_positives = false_positives * query_mask
      true_positives = true_positives.sum(axis=-1)
      gt_positives = gt_positives.sum(axis=-1)
      false_positives = false_positives.sum(axis=-1)
      jaccard = true_positives / (gt_positives + false_positives + 1e-8)
      all_jaccard.append(jaccard)

    values = jnp.mean(jnp.stack(all_jaccard, axis=-1), axis=-1)

    return self.State.from_values(values=values)
