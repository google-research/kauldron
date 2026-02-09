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

"""Frame sampling ops."""

from __future__ import annotations

import abc
import dataclasses

import grain.tensorflow as grain
from kauldron.typing import typechecked  # pylint: disable=g-importing-member,g-multiple-import
import tensorflow as tf


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class _ConstantStrideSampling(grain.MapTransform):
  """Sample a clip with constant stride from a random position in a video."""

  input_key: str = 'images'
  output_key: str | None = None
  stride: int = 1
  num_frames: int = 1
  indices_key: str | None = None
  num_clips: int = 1

  @typechecked
  def map(self, batch: dict[str, tf.Tensor]):
    frames = batch[self.input_key]
    vid_len = tf.shape(frames)[0]

    # sample the frames.
    frame_inds = self._get_frame_indices(vid_len)

    # Pad the video clip with the last frame if needed.
    frame_inds = tf.where(
        tf.greater(frame_inds, vid_len - 1),
        vid_len - 1,
        frame_inds,
    )

    # Add the sampled video clip to the batch.
    if not self.output_key:
      output_key = self.input_key
    else:
      output_key = self.output_key
    batch[output_key] = tf.gather(frames, frame_inds)

    if self.indices_key is not None:
      batch[self.indices_key] = frame_inds

    return batch

  @abc.abstractmethod
  def _get_frame_indices(self, vid_len):
    raise NotImplementedError


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandomClip(_ConstantStrideSampling):
  """Sample a clip with constant stride from a random position in a video."""

  @typechecked
  def _get_frame_indices(self, vid_len):
    # Randomly sample the start frame.
    end_frame = vid_len - (self.num_frames * self.stride)
    if end_frame > 0:
      start_frame = tf.random.uniform(
          shape=(),
          minval=0,
          maxval=end_frame,
          dtype=tf.int32,
      )
    else:
      start_frame = 0

    # Sample the frames with constant stride.
    frame_inds = tf.range(
        start_frame, start_frame + self.num_frames * self.stride, self.stride
    )
    frame_inds = tf.reshape(frame_inds, [self.num_frames])

    return frame_inds


class FirstClip(_ConstantStrideSampling):
  """Sample the first clip from a video with a constant stride."""

  def _get_frame_indices(self, vid_len):
    end_frame = self.num_frames * self.stride
    frame_inds = tf.range(0, end_frame, self.stride)
    frame_inds = tf.reshape(frame_inds, [self.num_frames])

    return frame_inds


class MiddleClip(_ConstantStrideSampling):
  """Sample the middle clip from a video with a constant stride."""

  def _get_frame_indices(self, vid_len):
    clip_dur = self.num_frames * self.stride
    diff = tf.maximum(0, vid_len - clip_dur)
    start_frame = diff // 2
    frame_inds = tf.cast(
        tf.range(
            start_frame,
            start_frame + (self.num_frames * self.stride),
            self.stride,
        ),
        dtype=tf.int32,
    )
    frame_inds = tf.reshape(frame_inds, [self.num_frames])
    return frame_inds


class MultiClip(_ConstantStrideSampling):
  """Sample multiple clips from a video with a constant stride."""

  def _get_frame_indices(self, vid_len):
    if self.num_clips < 2:
      raise ValueError(f'num_clips must be > 1: {self.num_clips}')

    max_offset = tf.maximum(0, vid_len - self.stride * self.num_frames)
    offsets = tf.linspace(0.0, tf.cast(max_offset, tf.float32), self.num_clips)
    offsets = tf.cast(offsets, tf.int32)

    frame_inds = tf.range(0, self.stride * self.num_frames, delta=self.stride)
    frame_inds = frame_inds[None, :] + offsets[:, None]
    frame_inds = tf.reshape(frame_inds, [self.num_frames * self.num_clips])
    return frame_inds
