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

"""Evaluation metrics for relative pose estimation."""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional

import flax
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from kauldron import kd
from kauldron import kontext
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import visu3d as v3d  # pytype: disable=import-error


def quaternion_to_rotation_matrix(quat: Float["*B 4"]) -> Float["*B 3 3"]:
  """Converts a quaternion to a rotation matrix."""
  return Rotation.from_quat(quat).as_matrix()


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RelativeRotationError(kd.metrics.Metric):
  """Computes the Relative Rotation Error.

  mode = "accuracy":
    The accuracy under a certain angular threshold will be reported.
    (as per the RRA metric from https://arxiv.org/abs/2306.15667).
  mode = "mean":
    The mean angular error will be reported.
  """

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  mode: Literal["mean", "accuracy"] = "accuracy"
  threshold_degs: float = 15.0
  rotation_as_quaternion: bool = False

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      preds: Float["*B T x y"],
      targets: Float["*B T x y"],
      mask: Optional[Bool["*B T x y"] | Float["*B T x y"]] = None,
  ) -> RelativeRotationError.State:
    if self.rotation_as_quaternion:
      preds = quaternion_to_rotation_matrix(preds[..., 0:4, 0])
      targets = quaternion_to_rotation_matrix(targets[..., 0:4, 0])

    rel_pose = (
        v3d.Transform.from_matrix(preds)
        @ v3d.Transform.from_matrix(targets).inv
    )
    delta_rot_matrix = rel_pose.R
    delta_angle = v3d.math.rot_to_rad(delta_rot_matrix) * 180 / jnp.pi

    if self.mode == "accuracy":
      values = delta_angle < self.threshold_degs
    else:
      values = delta_angle

    return self.State.from_values(values=values, mask=mask)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RelativeTranslationError(kd.metrics.Metric):
  """Computes the (angular) Relative Translation Error.

  *Note that due to scaling ambiguity, the translation error is assessed by the
  angle between the vectors, not the distance*

  mode = "accuracy":
    The accuracy under a certain angular threshold will be reported.
    (as per the RTA metric from https://arxiv.org/abs/2306.15667).
  mode = "mean":
    The mean translation angular error will be reported.
  """

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  mode: Literal["mean", "accuracy", "similarity"] = "accuracy"
  threshold_degs: float = 15.0
  rotation_as_quaternion: bool = False

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      preds: Float["*B Ts x y"],
      targets: Float["*B T x y"],
      mask: Optional[Bool["*B Ts x y"] | Float["*B Ts x y"]] = None,
      eps: float = 1e-8,
  ) -> RelativeTranslationError.State:
    if self.rotation_as_quaternion:
      transl_preds = preds[..., 4:, 0]
      transl_targets = targets[..., 4:, 0]
    else:
      transl_preds = preds[..., 3]
      transl_targets = targets[..., 3]
    # Normalize.
    transl_targets /= jnp.clip(
        jnp.linalg.norm(transl_targets, axis=-1, keepdims=True), min=eps
    )
    transl_preds /= jnp.clip(
        jnp.linalg.norm(transl_preds, axis=-1, keepdims=True), min=eps
    )

    dot_prod = jnp.sum(transl_preds * transl_targets, axis=-1)

    # Compute angles.
    transl_angle = jnp.acos(dot_prod) * 180 / jnp.pi

    if self.mode == "similarity":
      values = dot_prod
    elif self.mode == "accuracy":
      values = transl_angle < self.threshold_degs
    else:
      values = transl_angle

    return self.State.from_values(values=values, mask=mask)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PoseEndPointError(kd.metrics.Metric):
  """Computes the Pose End Point Error metric.

  This metric jointly evaluates the translation and rotation error by
  transforming a fixed grid of points by the predicted pose and comparing
  to the target pose. It is a 3D extension of the 2D metric defined in eq (2) in
  https://arxiv.org/pdf/1703.05593.
  """

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  _x_min: float = -5.0
  _x_max: float = 5.0
  _y_min: float = -5.0
  _y_max: float = 5.0
  _z_min: float = 0.0
  _z_max: float = 10.0
  _n_points: int = 2

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      preds: Float["*B T 3 4"],
      targets: Float["*B T 3 4"],
      mask: Optional[Bool["*B T 3 4"] | Float["*B T 3 4"]] = None,
  ) -> PoseEndPointError.State:
    x, y, z = jnp.meshgrid(
        jnp.linspace(self._x_min, self._x_max, self._n_points),
        jnp.linspace(self._y_min, self._y_max, self._n_points),
        jnp.linspace(self._z_min, self._z_max, self._n_points),
    )
    points_3d = jnp.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)
    b_star = preds.shape[:-2]
    preds = preds.reshape(-1, 3, 4)
    targets = targets.reshape(-1, 3, 4)
    points_3d_transformed_gt = (
        targets[:, :3, :3] @ points_3d + targets[:, :3, 3:]
    )
    points_3d_transformed_pred = preds[:, :3, :3] @ points_3d + preds[:, :3, 3:]
    epe = jnp.linalg.norm(
        points_3d_transformed_gt - points_3d_transformed_pred, axis=-2
    ).mean(axis=-1)
    epe = epe.reshape(b_star)

    return self.State.from_values(values=epe, mask=mask)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class PoseMeanAverageAccuracy(kd.metrics.Metric):
  """Calculate the Mean Average Accuracy (mAA) or (aka AUC) for the given error arrays using Jax.

  Originally from
  https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/util/metric.py

  """

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  threshold: int = 30
  rotation_as_quaternion: bool = False

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      preds: Float["*B T x y"],
      targets: Float["*B T x y"],
      mask: Optional[Bool["*B T x y"] | Float["*B T x y"]] = None,
      eps: float = 1e-8,
  ) -> PoseMeanAverageAccuracy.State:
    # get rotation error in degrees
    if self.rotation_as_quaternion:
      rot_preds = quaternion_to_rotation_matrix(preds[..., 0:4, 0])
      rot_targets = quaternion_to_rotation_matrix(targets[..., 0:4, 0])
    else:
      rot_preds = preds
      rot_targets = targets

    rel_pose = (
        v3d.Transform.from_matrix(rot_preds)
        @ v3d.Transform.from_matrix(rot_targets).inv
    )
    delta_rot_matrix = rel_pose.R
    delta_angle_rot = v3d.math.rot_to_rad(delta_rot_matrix) * 180 / jnp.pi

    # get translation error in degrees
    if self.rotation_as_quaternion:
      transl_preds = preds[..., 4:, 0]
      transl_targets = targets[..., 4:, 0]
    else:
      transl_preds = preds[..., 3]
      transl_targets = targets[..., 3]
    # Normalize.
    transl_targets /= jnp.clip(
        jnp.linalg.norm(transl_targets, axis=-1, keepdims=True), min=eps
    )
    transl_preds /= jnp.clip(
        jnp.linalg.norm(transl_preds, axis=-1, keepdims=True), min=eps
    )

    dot_prod = jnp.sum(transl_preds * transl_targets, axis=-1)
    delta_angle_transl = jnp.acos(dot_prod) * 180 / jnp.pi

    # get mAA / AUC
    error_matrix = jnp.stack((delta_angle_rot, delta_angle_transl), axis=1)
    max_errors = jnp.max(error_matrix, axis=1)  # [B, T]
    hist_fn = lambda x: jnp.histogram(
        x, bins=self.threshold + 1, range=[0, self.threshold]
    )[0]

    histogram = jax.vmap(hist_fn)(max_errors)  # [B, N_bins]
    num_pairs = float(max_errors.shape[1])
    normalized_histogram = histogram / num_pairs
    maa = jnp.mean(jnp.cumsum(normalized_histogram, axis=1), axis=1)  # [B]
    return self.State.from_values(values=maa)
