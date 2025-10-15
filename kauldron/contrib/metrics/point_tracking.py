# Copyright 2025 The kauldron Authors.
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

from collections.abc import Sequence
import dataclasses
from typing import Literal, Optional

import einops
import flax
import jax.numpy as jnp
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Array, Float, Shape, typechecked  # pylint: disable=g-multiple-import,g-importing-member


def get_evaluation_frames(
    query_frame: Float["*B Q"],
    num_frames: int,
    query_mode: Literal["first", "strided"],
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
  evaluation_mask: Optional[kontext.Key] = None  # e.g. "batch.evaluation_mask"
  query_frame: Optional[kontext.Key] = None  # e.g. "batch.query_frame"
  query_mask: Optional[kontext.Key] = None  # e.g. "batch.query_mask"
  query_mode: Literal["first", "strided"] = "strided"
  axis_order: Literal["BQTC", "BTQC"] = "BQTC"

  @flax.struct.dataclass
  class State(metrics.AutoState):
    values: Float["*"] | tuple[Float["*"], ...] = metrics.concat_field()

    def compute(self):
      return jnp.array(self.values).mean()

  @typechecked
  def get_state(
      self,
      pred_visible: Float["*B M N 1"],
      gt_visible: Float["*B M N 1"],
      evaluation_mask: Optional[Float["*B M N"]] = None,
      query_frame: Optional[Float["*B Q"]] = None,
      query_mask: Optional[Float["*B Q"]] = None,
  ) -> TapOcclusionAccuracy.State:
    """Computes the occlusion accuracy based on predicted and ground truth.

    Args:
      pred_visible: A tensor representing the predicted visibility of points
        with shape [..., M, N, 1], where M is the number of points, and N is the
        number of frames. The last dimension is squeezed and converted to
        boolean values.
      gt_visible: A tensor representing the ground truth visibility of points
        with shape [..., M, N, 1]. The last dimension is squeezed and converted
        to boolean values.
      evaluation_mask: An optional binary mask with shape [..., M, N] that
        specifies which points and frames should be considered for evaluation.
      query_frame: An optional tensor that specifies query frames. This is
        deprecated and retained for compatibility with older codebases.
      query_mask: An optional tensor that specifies a mask for query frames.
        This is also deprecated and retained for compatibility purposes.

    Returns:
      TapOcclusionAccuracy.State: A state object containing the computed
      occlusion accuracy for each batch.

    Notes:
      - The occlusion accuracy is calculated as the ratio of correct matches
        between the predicted and ground truth visibility tensors.
      - An evaluation mask can be used to exclude certain points or frames
        from the calculation.
      - If axis_order is set to "BTQC", the M and N dimensions of the
        visibility tensors and evaluation mask are swapped before the
        calculation.
      - The accuracy is computed by counting the number of matching entries
        where both the predicted and ground truth values are equal, divided
        by the total valid entries as defined by the evaluation mask.
      - A small constant (1e-8) is added to avoid division by zero when
        calculating the accuracy.
    """
    pred_visible = pred_visible.squeeze(-1).astype(bool)
    gt_visible = gt_visible.squeeze(-1).astype(bool)

    if self.axis_order == "BTQC":
      if query_frame is None:  # Assume query from first frame if not provided
        query_frame = jnp.zeros(Shape("*B N"))

      pred_visible = einops.rearrange(pred_visible, "... M N -> ... N M")
      gt_visible = einops.rearrange(gt_visible, "... M N -> ... N M")
      if evaluation_mask is None:
        evaluation_frames = get_evaluation_frames(
            query_frame, Shape("M")[0], self.query_mode
        )
      else:
        evaluation_frames = einops.rearrange(
            evaluation_mask.astype(bool), "... M N -> ... N M"
        )
    else:
      if query_frame is None:  # Assume query from first frame if not provided
        query_frame = jnp.zeros(Shape("*B M"))
      if evaluation_mask is None:
        evaluation_frames = get_evaluation_frames(
            query_frame, Shape("N")[0], self.query_mode
        )
      else:
        evaluation_frames = evaluation_mask.astype(bool)

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    values = jnp.equal(pred_visible, gt_visible) & evaluation_frames
    values = values.sum(axis=-1)
    count = evaluation_frames.sum(axis=-1)
    if query_mask is not None:
      values = values * query_mask
      count = count * query_mask
    values = values.sum(axis=-1) / (count.sum(axis=-1) + 1e-8)

    return self.State(values=values)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TapPositionAccuracy(metrics.Metric):
  """Position accuracy for visible points only."""

  pred_tracks: kontext.Key = kontext.REQUIRED  # e.g. "pred.tracks"
  gt_tracks: kontext.Key = kontext.REQUIRED  # e.g. "batch.target_points"
  gt_visible: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  evaluation_mask: Optional[kontext.Key] = None  # e.g. "batch.evaluation_mask"
  query_frame: Optional[kontext.Key] = None  # e.g. "batch.query_frame"
  query_mask: Optional[kontext.Key] = None  # e.g. "batch.query_mask"
  query_mode: Literal["first", "strided"] = "strided"
  axis_order: Literal["BQTC", "BTQC"] = "BQTC"
  # pixel radius to compute accuracy
  thresholds: Sequence[float] = (1 / 256, 2 / 256, 4 / 256, 8 / 256, 16 / 256)

  @flax.struct.dataclass
  class State(metrics.AutoState):
    values: Float["*"] | tuple[Float["*"], ...] = metrics.concat_field()

    def compute(self):
      return jnp.array(self.values).mean()

  @typechecked
  def get_state(
      self,
      pred_tracks: Float["*B M N 2"],
      gt_tracks: Float["*B M N 2"],
      gt_visible: Float["*B M N 1"],
      evaluation_mask: Optional[Float["*B M N"]] = None,
      query_frame: Optional[Float["*B Q"]] = None,
      query_mask: Optional[Float["*B Q"]] = None,
  ) -> TapPositionAccuracy.State:
    """Computes the position accuracy for visible points based on thresholds.

    Args:
      pred_tracks (Float["*B M N 2"]): Predicted track positions, with shape
        [..., M, N, 2], where M is the number of points and N is the number of
        frames. Each entry represents a 2D predicted point (x, y).
      gt_tracks (Float["*B M N 2"]): Ground truth track positions, with the same
        shape as pred_tracks, representing the 2D ground truth points.
      gt_visible (Float["*B M N 1"]): Ground truth visibility tensor, with shape
        [..., M, N, 1]. This tensor is squeezed and converted to boolean values,
        indicating whether points are visible.
      evaluation_mask (Optional[Float["*B M N"]]): A binary mask that specifies
        which points and frames should be considered for evaluation.
      query_frame (Optional[Float["*B Q"]]): Deprecated argument for query
        frames, retained for compatibility with legacy code.
      query_mask (Optional[Float["*B Q"]]): Deprecated argument for query masks,
        retained for compatibility with legacy code.

    Returns:
      TapPositionAccuracy.State: A state object containing the computed
      position accuracy across various distance thresholds.

    Notes:
      - The method calculates accuracy by checking how many predicted points
        fall within a series of pixel distance thresholds from the corresponding
        ground truth points.
      - The `thresholds` attribute defines the distance thresholds used to
        evaluate accuracy, and the final result is an average over all
        thresholds.
      - Only points visible in the ground truth and passing the evaluation
        mask are included in the accuracy computation.
      - The evaluation mask can be used to exclude certain points or frames
        from the evaluation process.
      - If the axis order is set to "BTQC", the M and N dimensions of the
        tracks and visibility tensors, along with the evaluation mask, are
        rearranged before computation.
      - The position accuracy is computed by averaging the fraction of points
        falling within the distance thresholds across all visible ground
        truth points.
    """
    gt_visible = gt_visible.squeeze(-1).astype(bool)

    if self.axis_order == "BTQC":
      if query_frame is None:  # Assume query from first frame if not provided
        query_frame = jnp.zeros(Shape("*B N"))

      pred_tracks = einops.rearrange(pred_tracks, "... M N C -> ... N M C")
      gt_tracks = einops.rearrange(gt_tracks, "... M N C -> ... N M C")
      gt_visible = einops.rearrange(gt_visible, "... M N -> ... N M")
      if evaluation_mask is None:
        evaluation_frames = get_evaluation_frames(
            query_frame, Shape("M")[0], self.query_mode
        )
      else:
        evaluation_frames = einops.rearrange(
            evaluation_mask.astype(bool), "... M N -> ... N M"
        )
    else:
      if query_frame is None:  # Assume query from first frame if not provided
        query_frame = jnp.zeros(Shape("*B M"))
      if evaluation_mask is None:
        evaluation_frames = get_evaluation_frames(
            query_frame, Shape("N")[0], self.query_mode
        )
      else:
        evaluation_frames = evaluation_mask.astype(bool)

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

    return self.State(values=values)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TapAverageJaccard(metrics.Metric):
  """Average Jaccard considering both location and occlusion accuracy."""

  pred_tracks: kontext.Key = kontext.REQUIRED  # e.g. "pred.tracks"
  pred_visible: kontext.Key = kontext.REQUIRED  # e.g. "pred.visible"
  gt_tracks: kontext.Key = kontext.REQUIRED  # e.g. "batch.target_points"
  gt_visible: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  evaluation_mask: Optional[kontext.Key] = None  # e.g. "batch.evaluation_mask"
  query_frame: Optional[kontext.Key] = None  # e.g. "batch.query_frame"
  query_mask: Optional[kontext.Key] = None  # e.g. "batch.query_mask"
  query_mode: Literal["first", "strided"] = "strided"
  axis_order: Literal["BQTC", "BTQC"] = "BQTC"

  # pixel radius to compute accuracy
  thresholds: Sequence[float] = (1 / 256, 2 / 256, 4 / 256, 8 / 256, 16 / 256)

  @flax.struct.dataclass
  class State(metrics.AutoState):
    values: Float["*"] | tuple[Float["*"], ...] = metrics.concat_field()

    def compute(self):
      return jnp.array(self.values).mean()

  @typechecked
  def get_state(
      self,
      pred_tracks: Float["*B M N 2"],
      pred_visible: Float["*B M N 1"],
      gt_tracks: Float["*B M N 2"],
      gt_visible: Array["*B M N 1"],
      evaluation_mask: Optional[Float["*B M N"]] = None,
      query_frame: Optional[Float["*B Q"]] = None,
      query_mask: Optional[Float["*B Q"]] = None,
  ) -> TapAverageJaccard.State:
    """Computes the Jaccard metric between predicted and ground truth tracks.

    Args:
      pred_tracks (Float["*B M N 2"]): Predicted track positions. Expected shape
        is [B, M, N, 2], where: - B is the batch size. - M is the number of
        frames or time steps. - N is the number of points to track. - 2
        corresponds to the spatial coordinates (e.g., x and y).
      pred_visible (Float["*B M N 1"]): Predicted visibility for each point.
        Shape is [B, M, N, 1], where 1 indicates whether each point is visible
        or not.
      gt_tracks (Float["*B M N 2"]): Ground truth track positions, with the same
        format as `pred_tracks`.
      gt_visible (Array["*B M N 1"]): Ground truth visibility for each point,
        with the same format as `pred_visible`.
      evaluation_mask (Optional[Float["*B M N"]]): A binary mask indicating
        which frames and points to evaluate.
      query_frame (Optional[Float["*B Q"]], optional): Deprecated in this
        computation but can be included for compatibility with other methods
        using a query frame.
      query_mask (Optional[Float["*B Q"]], optional): Deprecated in this
        computation but provided for compatibility with methods requiring query
        masks.

    Returns:
      TapAverageJaccard.State: A state object containing the average Jaccard
      metric values across multiple thresholds. The Jaccard metric is computed
      as the ratio of true positives to the sum of true positives, false
      positives, and false negatives, averaged across all frames and points.

    Notes:
      - For points to be considered in the Jaccard calculation, both the
        prediction and ground truth must indicate visibility.
      - The evaluation mask can be used to exclude certain points or frames
        from being evaluated.
      - If the axis order is set to "BTQC", the tracks and visibility tensors,
        as well as the evaluation mask, are rearranged to swap the M and N
        dimensions before performing the computations.
      - Refer to the TAP-Vid paper for details on the metric computation and
        its application.
    """
    pred_visible = pred_visible.squeeze(-1).astype(bool)
    gt_visible = gt_visible.squeeze(-1).astype(bool)

    if self.axis_order == "BTQC":
      if query_frame is None:  # Assume query from first frame if not provided
        query_frame = jnp.zeros(Shape("*B N"))

      pred_tracks = einops.rearrange(pred_tracks, "... M N C -> ... N M C")
      pred_visible = einops.rearrange(pred_visible, "... M N -> ... N M")
      gt_tracks = einops.rearrange(gt_tracks, "... M N C -> ... N M C")
      gt_visible = einops.rearrange(gt_visible, "... M N -> ... N M")
      if evaluation_mask is None:
        evaluation_frames = get_evaluation_frames(
            query_frame, Shape("M")[0], self.query_mode
        )
      else:
        evaluation_frames = einops.rearrange(
            evaluation_mask.astype(bool), "... M N -> ... N M"
        )
    else:
      if query_frame is None:  # Assume query from first frame if not provided
        query_frame = jnp.zeros(Shape("*B M"))
      if evaluation_mask is None:
        evaluation_frames = get_evaluation_frames(
            query_frame, Shape("N")[0], self.query_mode
        )
      else:
        evaluation_frames = evaluation_mask.astype(bool)

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

    return self.State(values=values)
