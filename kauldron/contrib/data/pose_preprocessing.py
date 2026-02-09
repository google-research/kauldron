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

"""Pose preprocessing ops."""

from __future__ import annotations

import dataclasses

from kauldron import kd
from kauldron.typing import TfArray, typechecked  # pylint: disable=g-importing-member,g-multiple-import
import tensorflow as tf
from tensorflow_graphics.geometry.transformation import quaternion  # pytype: disable=import-error
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d  # pytype: disable=import-error


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class TartanPoseToRT(kd.data.ElementWiseTransform):
  """Converts from TartanAir pose convention to standard SfM [RT] convention."""

  def _trans_quat_to_RT(self, trans_quat: TfArray["7"]) -> TfArray["3 4"]:
    """Convert single trans_quat element to [RT]."""
    trans = trans_quat[:3, None]
    quat = trans_quat[3:]

    rot = rotation_matrix_3d.from_quaternion(quat)

    # Convert coordinate convention from NED to OpenCV
    rot = tf.gather(rot, [1, 2, 0], axis=1)

    return tf.concat([rot, trans], axis=1)

  @typechecked
  def map_element(self, element: TfArray["T 7"]) -> TfArray["T 3 4"]:
    return tf.map_fn(self._trans_quat_to_RT, element)  # apply to all times


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RelativePoses(kd.data.ElementWiseTransform):
  """Computes relative poses across video with given stride.

  Attributes:
    stride: The stride to use for computing relative poses.
    normalize_translation: Whether to normalize the translation vector to unit
      norm.
    cam_to_world_poses: Whether the input poses are camera-to-world or
      world-to-camera.
    relative_to_first_frame: Whether to compute relative poses relative to the
      first frame or relative to the previous frame.
    input_rotation_as_quaternion: input poses are quaternion or matrix.
    output_rotation_as_quaternion: output poses are quaternion or matrix.
  """

  stride: int
  normalize_translation: bool = False
  cam_to_world_poses: bool = True
  relative_to_first_frame: bool = False
  input_rotation_as_quaternion: bool = False
  output_rotation_as_quaternion: bool = False

  def _invert_pose(self, pose: TfArray["T x y"]):
    inv_rot = tf.transpose(pose[..., :3, :3], perm=[0, 2, 1])
    inv_trans = -inv_rot @ pose[..., :3, 3:]
    return tf.concat([inv_rot, inv_trans], axis=-1)

  def _compose_poses(
      self, pose_left: TfArray["T x y"], pose_right: TfArray["T x y"]
  ) -> TfArray["T x y"]:
    rots_left = pose_left[..., :3, :3]
    rots_right = pose_right[..., :3, :3]
    transl_left = pose_left[..., :3, 3:]
    transl_right = pose_right[..., :3, 3:]
    rot = rots_left @ rots_right
    transl = rots_left @ transl_right + transl_left
    if self.normalize_translation:
      eps = 1e-7
      transl = transl / (tf.norm(transl, axis=-2, keepdims=True) + eps)
    if self.output_rotation_as_quaternion:
      quat = quaternion.from_rotation_matrix(rot)  # x y z w
      return tf.concat([quat[..., None], transl], axis=-2)  # [T, 7, 1]
    else:
      return tf.concat([rot, transl], axis=-1)  # [T, 3, 4]

  @typechecked
  def map_element(
      self,
      element: TfArray["T X Y"],
  ) -> TfArray["Ts Xs Ys"]:
    valid_mask = None

    if self.input_rotation_as_quaternion:  # [T,7,1]
      rot = rotation_matrix_3d.from_quaternion(element[:, 0:4, 0])  # [T,3,3]
      element = tf.concat([rot, element[:, 4::, :]], axis=-1)  # [T,3,4]

    if self.stride == -1:
      # self.stride == -1 is a special mode,
      # it means getting dense pairwise poses with shape [T^2, 3, 4]
      # order is important such that the prediction can be
      # masked with a tril mask for dense pairwise poses, e.g.:
      # [---], [---], [---], [---]
      # [0,1], [---], [---], [---]
      # [0,2], [1,2], [---], [---]
      # [0,3], [1,3], [2,3], [---]
      # then the valid_mask is:
      # tf.experimental.numpy.tril(tf.ones((T, T), dtype=tf.float32), k=-1)

      # or if we want to consider bi-directional pairwise poses:
      # [---], [1,0], [2,0], [3,0]
      # [0,1], [---], [2,1], [3,1]
      # [0,2], [1,2], [---], [3,2]
      # [0,3], [1,3], [2,3], [---]
      # then the valid_mask is:
      # 1 - tf.eye(T, dtype=tf.float32)
      T = element.shape[0]  # pylint: disable=invalid-name
      source_poses = tf.repeat(element[None, :], T, axis=0)
      target_poses = tf.repeat(element[:, None], T, axis=1)
      source_poses = tf.reshape(source_poses, shape=(-1, 3, 4))
      target_poses = tf.reshape(target_poses, shape=(-1, 3, 4))
      valid_mask = (
          tf.reshape(
              1 - tf.eye(T, dtype=tf.float32),
              shape=(-1,),
          )[:, None, None]
          > 0
      )
    else:
      element = element[:: self.stride]
      if self.relative_to_first_frame:
        source_poses = element[:1]
        target_poses = element
      else:
        source_poses = element[:-1]
        target_poses = element[1:]
    if self.cam_to_world_poses:
      output_poses = self._compose_poses(
          self._invert_pose(source_poses), target_poses
      )
    else:
      output_poses = self._compose_poses(
          source_poses, self._invert_pose(target_poses)
      )

    if self.stride == -1:
      if self.output_rotation_as_quaternion:
        empty_pose = tf.expand_dims(
            tf.constant([0, 0, 0, 1, 0, 0, 0], dtype=tf.float32), axis=-1
        )
      else:
        empty_pose = tf.constant(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=tf.float32
        )
      output_poses = tf.where(valid_mask, output_poses, empty_pose[None, :])
    return output_poses
