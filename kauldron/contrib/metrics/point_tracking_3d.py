# Copyright 2026 The kauldron Authors.
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

"""Evaluation metrics for 3D point tracking.

Differs from the 2D point tracking metric in that:
1. It uses and expects camera intrinsics, scaled to 256x256 resolution size
2. Requires a choice of depth scaling method (median or reproduce_2d)
3. The point error threshold is computed from the physical metric distance,
   rather than the pixel distance.

"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
from typing import Literal, Optional

import einops
import flax
import jax.numpy as jnp
from kauldron import kontext
from kauldron import metrics
from kauldron.contrib.metrics import point_tracking
from kauldron.typing import Bool, Dim, Float, Int, Shape, set_shape, typechecked  # pylint: disable=g-multiple-import,g-importing-member

# pytype: disable=attribute-error

PIXEL_TO_FIXED_METRIC_THRESH = {
    1: 0.01,
    2: 0.04,
    4: 0.16,
    8: 0.64,
    16: 2.56,
}


@typechecked
def gather_queried_tracks(
    preds: Float["B T H W C"],
    query_points: Int["B K 2"] | Float["B K 2"],
    use_normalized_coordinates: bool,
):
  """Gather queried tracks from dense HxW predictions."""
  if use_normalized_coordinates:
    query_points = query_points * jnp.array([Dim("W"), Dim("H")])
  indices = query_points.astype(jnp.int32)
  gathered = gather_points_jax(preds, indices)
  return gathered


@typechecked
def gather_points_jax(tensor: Float["B T H W C"], indices: Int["B K 2"]):
  """Gathers points from tensor N using batched indices M in JAX.

  Args:
      tensor: A JAX tensor of shape [B, T, H, W, C].
      indices: A JAX tensor of shape [B, K, 2], the last dimension represents
        (x, y) indices into the H and W dimensions of N.

  Returns:
      A JAX tensor of shape [B, T, K, C] containing the gathered values.
  """
  # Expand M to include the time dimension (T)
  ind_expanded = jnp.expand_dims(indices, axis=1)  # [B, 1, K, 2]
  ind_expanded = jnp.tile(ind_expanded, [1, Dim("T"), 1, 1])  # [B, T, K, 2]

  # Get the x and y indices
  x_indices = ind_expanded[:, :, :, 0]  # [B, T, K]
  y_indices = ind_expanded[:, :, :, 1]  # [B, T, K]

  # Create batch indices.
  batch_indices = jnp.arange(Dim("B"))
  batch_indices = batch_indices[:, None, None]
  batch_indices = jnp.broadcast_to(batch_indices, Shape("B T K"))

  # Create time indices
  time_indices = jnp.arange(Dim("T"))
  time_indices = time_indices[None, :, None]
  time_indices = jnp.broadcast_to(time_indices, Shape("B T K"))
  # Stack indices to gather
  indices = jnp.stack(
      [batch_indices, time_indices, y_indices, x_indices], axis=-1
  )  # [B, T, K, 4]

  # Gather from N using batch and spatial indices
  output = tensor[
      indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
  ]  # [B, T, K, C]

  return output


def image_to_camera_3d(
    points: Float["*B K 2"], depth: Float["*B K 1"], intrinsics: Float["*B 4"]
):
  # image to camera coords.
  intrinsics = intrinsics[:, None, :]
  f_x, f_y, c_x, c_y = jnp.split(intrinsics, 4, axis=-1)
  x_cam = (points[..., 0] - c_x) * depth / f_x
  y_cam = (points[..., 1] - c_y) * depth / f_y
  return jnp.stack(
      [x_cam, y_cam, depth],
      axis=-1
  )


def get_scale_factor(
    gt_visible: Bool["*B Q T"],
    pred_visible: Bool["*B Q T"],
    pred_tracks: Float["*B Q T 3"],
    gt_tracks: Float["*B Q T 3"],
    depth_scaling: Literal["median", "reproduce_2d"],
):
  """Computes the scale factor between predicted and ground truth tracks."""
  pred_norms = jnp.sqrt(
      jnp.maximum(1e-12, jnp.sum(jnp.square(pred_tracks), axis=-1)))
  gt_norms = jnp.sqrt(
      jnp.maximum(1e-12, jnp.sum(jnp.square(gt_tracks), axis=-1)))

  gt_occluded = jnp.logical_not(gt_visible)
  pred_occluded = jnp.logical_not(pred_visible)
  either_occluded = jnp.logical_or(gt_occluded, pred_occluded)
  nan_mat = jnp.full(pred_norms.shape, jnp.nan)
  pred_norms = jnp.where(either_occluded, nan_mat, pred_norms)
  gt_norms = jnp.where(either_occluded, nan_mat, gt_norms)

  if depth_scaling == "median":
    scale_factor = jnp.nanmedian(
        gt_norms, axis=(-2, -1), keepdims=True
    ) / jnp.nanmedian(pred_norms, axis=(-2, -1), keepdims=True)

  elif depth_scaling == "reproduce_2d":
    scale_factor = gt_tracks[..., -1:] / pred_tracks[..., -1:]

  else:
    raise ValueError(f"Unknown scaling method: {depth_scaling}")

  return scale_factor[..., None]


def get_pointwise_threshold_multiplier(
    gt_tracks: Float["*B Q T 3"],
    intrinsics_params: Float["*B 4"]
):
  mean_focal_length = jnp.sqrt(
      intrinsics_params[..., 0] * intrinsics_params[..., 1]
  )
  return gt_tracks[..., -1] / mean_focal_length[..., jnp.newaxis, jnp.newaxis]


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Tap3DPositionAccuracy(metrics.Metric):
  """Position accuracy for visible points only."""

  pred_tracks: kontext.Key = kontext.REQUIRED  # e.g. "pred.tracks"
  pred_visible: kontext.Key = kontext.REQUIRED  # e.g. "pred.visible"
  gt_tracks: kontext.Key = kontext.REQUIRED  # e.g. "batch.target_points"
  gt_visible: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  evaluation_mask: Optional[kontext.Key] = None  # e.g. "batch.evaluation_mask"
  intrinsics: kontext.Key = kontext.REQUIRED  # e.g. "batch.intrinsics"
  gt_tracks_2d: Optional[kontext.Key] = None
  pred_tracks_2d: Optional[kontext.Key] = None
  offline_depth_predictions: Optional[kontext.Key] = None
  depth_scaling: Literal["median", "reproduce_2d"] = "median"
  query_mode: Literal["first", "strided"] = "strided"
  axis_order: Literal["BQTC", "BTQC"] = "BQTC"
  use_fixed_metric_threshold: bool = False
  # pixel radius (projected to 3D) to compute accuracy
  thresholds: Sequence[float] = (1, 2, 4, 8, 16)
  use_offline_depth_predictions: bool = False
  use_gt_tracks_for_indexing_into_offline_depth_predictions: bool = False
  track_max_resolution: tuple[int, int] = (256, 256)
  use_normalized_coordinates: bool = False

  @flax.struct.dataclass
  class State(metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred_tracks: Float["*B M N 3"],
      pred_visible: Float["*B M N 1"] | Bool["*B M N 1"],
      gt_tracks: Float["*B M N 3"],
      gt_visible: Float["*B M N 1"] | Bool["*B M N 1"],
      intrinsics: Float["*B 4"],
      evaluation_mask: Optional[Float["*B M N"] | Bool["*B M N"]] = None,
      offline_depth_predictions: Optional[
          Float["*B T H W 1"] | Float["*B M N 1"]
      ] = None,
      gt_tracks_2d: Optional[Float["*B M N 2"]] = None,
      pred_tracks_2d: Optional[Float["*B M N 2"]] = None,
  ) -> Tap3DPositionAccuracy.State:
    """Computes the 3D position accuracy b/w predicted and ground truth tracks.

    Note: The axis order (BQTC or BTQC) determines how the data is interpreted.
    For BQTC, M represents the number of queries and N represents the number of
    frames. For BTQC, M represents the number of frames and N represents the
    number of queries.

    Args:
      pred_tracks (Float["*B M N 3"]): Predicted track positions, with shape
        [..., M, N, 2], where N is the number of trajectories and M is the
        number of frames (or vice-versa, depending on axis_order). Each entry
        represents a 3D predicted point (x, y, z).
      pred_visible (Float["*B M N 1"]): Predicted occlusion tensor, with shape
        [..., M, N, 1]. This tensor is squeezed and converted to boolean values,
        indicating whether points are occluded.
      gt_tracks (Float["*B M N 3"]): Ground truth track positions, with the same
        shape as pred_tracks, representing the 3D ground truth points.
      gt_visible (Float["*B M N 1"]): Ground truth visibility tensor, with shape
        [..., M, N, 1]. This tensor is squeezed and converted to boolean values,
        indicating whether points are visible.
      intrinsics (Float["*B 4"]): Intrinsics matrix, with shape [..., 4],
        representing (fx, fy, cx, cy). Expects normalized coordinates if
        'use_normalized_coordinates' is True, otherwise expects pixel
        coordinates scaled to 'track_max_resolution'.
      evaluation_mask (Optional[Float["*B M N"]]): A binary mask that specifies
        which points and frames should be considered for evaluation.
      offline_depth_predictions: Predicted depth values. If shape is [..., T, H,
        W, 1], H, W correspond exactly to the video pixel size. If shape is
        [..., M, N, 1], then depth is already sparse and no gathering is needed.
        Used instead of the pred_tracks depth if
        self.use_offline_depth_predictions is True.
      gt_tracks_2d (Optional[Float["*B M N 2"]]): GT sparse track positions in
        2D pixel space, with shape [..., M, N, 2], where N is the number of
        trajectories and M is the number of frames. Each entry represents a 2D
        predicted point (x, y). Expects normalized coordinates if
        'use_normalized_coordinates' is True, otherwise expects pixel
        coordinates scaled to 'track_max_resolution'.
      pred_tracks_2d (Optional[Float["*B M N 2"]]): Predicted sparse tracks,
        similar to gt_tracks_2d.

    Returns:
      Tap3DPositionAccuracy.State: A state object containing the computed
      position accuracy across various distance thresholds.
    """
    pred_visible = pred_visible.squeeze(-1).astype(bool)
    gt_visible = gt_visible.squeeze(-1).astype(bool)

    if self.axis_order == "BTQC":
      set_shape("T Q", Shape("M N"))
      query_frame = jnp.zeros(Shape("*B Q"))
      pred_tracks = einops.rearrange(pred_tracks, "... T Q C -> ... Q T C")
      pred_visible = einops.rearrange(pred_visible, "... T Q -> ... Q T")
      gt_tracks = einops.rearrange(gt_tracks, "... T Q C -> ... Q T C")
      gt_visible = einops.rearrange(gt_visible, "... T Q -> ... Q T")
      if evaluation_mask is None:
        evaluation_frames = point_tracking.get_evaluation_frames(
            query_frame, Shape("T")[0], self.query_mode
        )
      else:
        evaluation_frames = einops.rearrange(
            evaluation_mask.astype(bool), "... T Q -> ... Q T"
        )
    else:
      set_shape("Q T", Shape("M N"))
      query_frame = jnp.zeros(Shape("*B Q"))
      if evaluation_mask is None:
        evaluation_frames = point_tracking.get_evaluation_frames(
            query_frame, Shape("T")[0], self.query_mode
        )
      else:
        evaluation_frames = evaluation_mask.astype(bool)

    if self.use_offline_depth_predictions:
      assert self.axis_order == "BTQC"
      if self.use_gt_tracks_for_indexing_into_offline_depth_predictions:
        tracks_2d = gt_tracks_2d
      else:
        tracks_2d = pred_tracks_2d

      if offline_depth_predictions.ndim == 5:
        # Dense depth map are provided as [..., T, H, W, 1]. We need to gather
        # the depth values at the corresponding pixel locations for each track.
        depth = einops.rearrange(
            offline_depth_predictions, "B T H W 1 -> (B T) 1 H W 1"
        )
        sparse_depth = gather_queried_tracks(
            depth,
            einops.rearrange(tracks_2d, "B T Q C -> (B T) Q C"),
            self.use_normalized_coordinates,
        )
        sparse_depth = einops.rearrange(
            sparse_depth, "(B T) 1 Q C -> B T Q C", B=tracks_2d.shape[0]
        )
      else:
        # Depth is already sparse.
        if self.axis_order == "BQTC":
          offline_depth_predictions = einops.rearrange(
              offline_depth_predictions, "... Q T C -> ... T Q C"
          )
        sparse_depth = offline_depth_predictions
      sparse_depth = sparse_depth.squeeze(-1)

      pred_tracks = image_to_camera_3d(
          tracks_2d[..., 0:2], sparse_depth, intrinsics
      )
      pred_tracks = einops.rearrange(pred_tracks, "... T Q C -> ... Q T C")

    scale_factor = get_scale_factor(
        gt_visible=gt_visible,
        pred_visible=pred_visible,
        pred_tracks=pred_tracks,
        gt_tracks=gt_tracks,
        depth_scaling=self.depth_scaling,
    )
    pred_tracks = pred_tracks * scale_factor

    if self.use_normalized_coordinates:
      # Pointwise threshold is computed from scaled intrinsics matching
      # evaluation resolution (256x256 for TAPVid3D). So if the intrinsics
      # are in normalized coordinates, we need to convert them to
      # unnormalized pixel coordinates.
      unnormalized_intrinsics = intrinsics * jnp.array([
          self.track_max_resolution[1],
          self.track_max_resolution[0],
          self.track_max_resolution[1],
          self.track_max_resolution[0],
      ])
    else:
      unnormalized_intrinsics = intrinsics

    all_frac_within = []
    for thresh in self.thresholds:

      if self.use_fixed_metric_threshold:
        pointwise_thresh = PIXEL_TO_FIXED_METRIC_THRESH[thresh]
      else:
        multiplier = get_pointwise_threshold_multiplier(
            gt_tracks, unnormalized_intrinsics
        )
        pointwise_thresh = thresh * multiplier

      # True positives are points that are within the threshold and where both
      # the prediction and the ground truth are listed as visible.
      within_dist = jnp.sum(
          jnp.square(pred_tracks - gt_tracks), axis=-1
      ) < jnp.square(pointwise_thresh)
      is_correct = jnp.logical_and(within_dist, gt_visible)

      # Compute the frac_within_threshold, which is the fraction of points
      # within the threshold among points that are visible in the ground truth,
      # ignoring whether they're predicted to be visible.
      count_correct = jnp.sum(is_correct & evaluation_frames, axis=-1)
      count_visible_points = jnp.sum(gt_visible & evaluation_frames, axis=-1)
      count_correct = count_correct.sum(axis=-1)
      count_visible_points = count_visible_points.sum(axis=-1)
      frac_correct = count_correct / (count_visible_points + 1e-8)
      all_frac_within.append(frac_correct)

    values = jnp.mean(jnp.stack(all_frac_within, axis=-1), axis=-1)

    return self.State.from_values(values=values)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Tap3DAverageJaccard(metrics.Metric):
  """3D Average Jaccard considering both location and occlusion accuracy."""

  pred_tracks: kontext.Key = kontext.REQUIRED  # e.g. "pred.tracks"
  pred_visible: kontext.Key = kontext.REQUIRED  # e.g. "pred.visible"
  gt_tracks: kontext.Key = kontext.REQUIRED  # e.g. "batch.target_points"
  gt_visible: kontext.Key = kontext.REQUIRED  # e.g. "batch.visible"
  evaluation_mask: Optional[kontext.Key] = None  # e.g. "batch.evaluation_mask"
  intrinsics: kontext.Key = kontext.REQUIRED  # e.g. "batch.intrinsics"
  gt_tracks_2d: Optional[kontext.Key] = None
  pred_tracks_2d: Optional[kontext.Key] = None
  offline_depth_predictions: Optional[kontext.Key] = None
  depth_scaling: Literal["median", "reproduce_2d"] = "median"
  query_mode: Literal["first", "strided"] = "strided"
  axis_order: Literal["BQTC", "BTQC"] = "BQTC"
  use_fixed_metric_threshold: bool = False
  # pixel radius (projected to 3D) to compute accuracy
  thresholds: Sequence[float] = (1, 2, 4, 8, 16)
  use_offline_depth_predictions: bool = False
  use_gt_tracks_for_indexing_into_offline_depth_predictions: bool = False
  track_max_resolution: tuple[int, int] = (256, 256)
  use_normalized_coordinates: bool = False

  @flax.struct.dataclass
  class State(metrics.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      pred_tracks: Float["*B M N 3"],
      pred_visible: Float["*B M N 1"] | Bool["*B M N 1"],
      gt_tracks: Float["*B M N 3"],
      gt_visible: Float["*B M N 1"] | Bool["*B M N 1"],
      intrinsics: Float["*B 4"],
      evaluation_mask: Optional[Float["*B M N"] | Bool["*B M N"]] = None,
      gt_tracks_2d: Optional[Float["*B M N 2"]] = None,
      pred_tracks_2d: Optional[Float["*B M N 2"]] = None,
      offline_depth_predictions: Optional[
          Float["*B T H W 1"] | Float["*B M N 1"]
      ] = None,
  ) -> Tap3DAverageJaccard.State:
    """Computes the Jaccard metric between predicted and ground truth tracks.

    Note: The axis order (BQTC or BTQC) determines how the data is interpreted.
    For BQTC, M represents the number of queries and N represents the number of
    frames. For BTQC, M represents the number of frames and N represents the
    number of queries.

    Args:
      pred_tracks (Float["*B M N 3"]): Predicted track positions, with shape
        [..., M, N, 2], where M is the number of trajectories and N is the
        number of frames (or vice-versa, depending on axis_order). Each entry
        represents a 3D predicted point (x, y, z).
      pred_visible (Float["*B M N 1"]): Predicted occlusion tensor, with shape
        [..., M, N, 1]. This tensor is squeezed and converted to boolean values,
        indicating whether points are occluded.
      gt_tracks (Float["*B M N 3"]): Ground truth track positions, with the same
        shape as pred_tracks, representing the 3D ground truth points.
      gt_visible (Float["*B M N 1"]): Ground truth visibility tensor, with shape
        [..., M, N, 1]. This tensor is squeezed and converted to boolean values,
        indicating whether points are visible.
      intrinsics (Float["*B 4"]): Intrinsics matrix, with shape [..., 4],
        representing (fx, fy, cx, cy). Expects normalized coordinates if
        'use_normalized_coordinates' is True, otherwise expects pixel
        coordinates scaled to 'track_max_resolution'.
      evaluation_mask (Optional[Float["*B M N"]]): A binary mask that specifies
        which points and frames should be considered for evaluation.
      gt_tracks_2d (Optional[Float["*B M N 2"]]): Ground truth sparse track
        positions in 2D pixel space, with shape [..., M, N, 2], where N is the
        number of trajectories and M is the number of frames. Each entry
        represents a 2D predicted point (x, y). Expects normalized coordinates
        if 'use_normalized_coordinates' is True, otherwise expects pixel
        coordinates scaled to 'track_max_resolution'.
      pred_tracks_2d (Optional[Float["*B M N 2"]]): Predicted sparse tracks,
        similar to gt_tracks_2d.
      offline_depth_predictions: Predicted depth values. If shape is [..., T, H,
        W, 1], H, W correspond exactly to the video pixel size. If shape is
        [..., M, N, 1], then depth is already sparse and no gathering is needed.
        Used instead of the pred_tracks depth if
        self.use_offline_depth_predictions is True.

    Returns:
      Tap3DAverageJaccard.State: A state object containing the average Jaccard
      metric values across multiple thresholds. The Jaccard metric is computed
      as the ratio of true positives to the sum of true positives, false
      positives, and false negatives, averaged across all frames and points.
    """
    pred_visible = pred_visible.squeeze(-1).astype(bool)
    gt_visible = gt_visible.squeeze(-1).astype(bool)

    if self.axis_order == "BTQC":
      set_shape("T Q", Shape("M N"))
      query_frame = jnp.zeros(Shape("*B Q"))
      pred_tracks = einops.rearrange(pred_tracks, "... T Q C -> ... Q T C")
      pred_visible = einops.rearrange(pred_visible, "... T Q -> ... Q T")
      gt_tracks = einops.rearrange(gt_tracks, "... T Q C -> ... Q T C")
      gt_visible = einops.rearrange(gt_visible, "... T Q -> ... Q T")
      if evaluation_mask is None:
        evaluation_frames = point_tracking.get_evaluation_frames(
            query_frame, Shape("T")[0], self.query_mode
        )
      else:
        evaluation_frames = einops.rearrange(
            evaluation_mask.astype(bool), "... T Q -> ... Q T"
        )
    else:
      set_shape("Q T", Shape("M N"))
      query_frame = jnp.zeros(Shape("*B Q"))
      if evaluation_mask is None:
        evaluation_frames = point_tracking.get_evaluation_frames(
            query_frame, Shape("T")[0], self.query_mode
        )
      else:
        evaluation_frames = evaluation_mask.astype(bool)

    if self.use_offline_depth_predictions:
      assert self.axis_order == "BTQC"
      if self.use_gt_tracks_for_indexing_into_offline_depth_predictions:
        tracks_2d = gt_tracks_2d
      else:
        tracks_2d = pred_tracks_2d

      if offline_depth_predictions.ndim == 5:
        # Dense depth map are provided as [..., T, H, W, 1]. We need to gather
        # the depth values at the corresponding pixel locations for each track.
        depth = einops.rearrange(
            offline_depth_predictions, "B T H W 1 -> (B T) 1 H W 1"
        )
        sparse_depth = gather_queried_tracks(
            depth,
            einops.rearrange(tracks_2d, "B T Q C -> (B T) Q C"),
            self.use_normalized_coordinates,
        )
        sparse_depth = einops.rearrange(
            sparse_depth, "(B T) 1 Q C -> B T Q C", B=tracks_2d.shape[0]
        )
      else:
        # Depth is already sparse.
        if self.axis_order == "BQTC":
          offline_depth_predictions = einops.rearrange(
              offline_depth_predictions, "... Q T C -> ... T Q C"
          )
        sparse_depth = offline_depth_predictions
      sparse_depth = sparse_depth.squeeze(-1)

      pred_tracks = image_to_camera_3d(
          tracks_2d[..., 0:2], sparse_depth, intrinsics
      )
      pred_tracks = einops.rearrange(pred_tracks, "... T Q C -> ... Q T C")

    scale_factor = get_scale_factor(
        gt_visible=gt_visible,
        pred_visible=pred_visible,
        pred_tracks=pred_tracks,
        gt_tracks=gt_tracks,
        depth_scaling=self.depth_scaling,
    )
    pred_tracks = pred_tracks * scale_factor

    if self.use_normalized_coordinates:
      # Pointwise threshold is computed from scaled intrinsics matching
      # evaluation resolution (256x256 for TAPVid3D). So if the intrinsics
      # are in normalized coordinates, we need to convert them to
      # unnormalized pixel coordinates.
      unnormalized_intrinsics = intrinsics * jnp.array([
          self.track_max_resolution[1],
          self.track_max_resolution[0],
          self.track_max_resolution[1],
          self.track_max_resolution[0],
      ])
    else:
      unnormalized_intrinsics = intrinsics

    all_jaccard = []
    for thresh in self.thresholds:

      if self.use_fixed_metric_threshold:
        pointwise_thresh = PIXEL_TO_FIXED_METRIC_THRESH[thresh]
      else:
        multiplier = get_pointwise_threshold_multiplier(
            gt_tracks, unnormalized_intrinsics
        )
        pointwise_thresh = thresh * multiplier

      # True positives are points that are within the threshold and where both
      # the prediction and the ground truth are listed as visible.
      within_dist = jnp.sum(
          jnp.square(pred_tracks - gt_tracks), axis=-1
      ) < jnp.square(pointwise_thresh)
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
      true_positives = true_positives.sum(axis=-1)
      gt_positives = gt_positives.sum(axis=-1)
      false_positives = false_positives.sum(axis=-1)
      jaccard = true_positives / (gt_positives + false_positives + 1e-8)
      all_jaccard.append(jaccard)

    values = jnp.mean(jnp.stack(all_jaccard, axis=-1), axis=-1)

    return self.State.from_values(values=values)
