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

"""GIF Video Summaries for TensorBoard.

Video metrics that return UInt8 5D arrays for encoding as animated GIFs.
Use with `GifVideoWriter` from `kauldron.contrib.train.gif_video_writer`.

Example usage:
  from kauldron.contrib.train import GifVideoWriter
  from kauldron.contrib.summaries import ShowVideosAsGif

  cfg.writer = GifVideoWriter()
  cfg.train_summaries["video"] = ShowVideosAsGif(
      videos="batch.video",
      in_vrange=(-1.0, 1.0),
  )
"""

from __future__ import annotations

import dataclasses

import flax
import jax.numpy as jnp
from kauldron import kd
from kauldron.typing import Num, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True)
class ShowVideosAsGif(kd.metrics.Metric):
  """Show video as animated GIF in TensorBoard.

  Returns UInt8 5D array that GifVideoWriter will encode as animated GIF.
  Use with `GifVideoWriter` from `kauldron.contrib.train.gif_video_writer`.

  Attributes:
    videos: Key to the videos to display [N, T, H, W, C].
    num_videos: Number of videos to collect and display.
    in_vrange: Optional value range to normalize to [0, 1].
    save_format: Output format: 'GIF', 'WEBP', or 'PNG'.
  """

  videos: kd.kontext.Key

  num_videos: int = 4
  in_vrange: tuple[float, float] | None = None
  save_format: str = "GIF"

  @flax.struct.dataclass
  class State(kd.metrics.AutoState["ShowVideosAsGif"]):
    """Collects videos."""

    videos: UInt8["n t h w #3"] = kd.metrics.truncate_field(
        num_field="parent.num_videos"
    )

    @typechecked
    def compute(self) -> UInt8["n t h w #3"]:
      return super().compute().videos

  @typechecked
  def get_state(
      self,
      videos: Num["n t h w #3"],
  ) -> "ShowVideosAsGif.State":
    if videos.shape[-1] == 1:
      videos = jnp.repeat(videos, 3, axis=-1)

    videos = jnp.asarray(videos, dtype=jnp.float32)
    videos = videos[: self.num_videos]

    if self.in_vrange is not None:
      vmin, vmax = self.in_vrange
      videos = jnp.clip(videos, vmin, vmax)
      videos = (videos - vmin) / (vmax - vmin)

    videos_uint8 = (jnp.clip(videos, 0.0, 1.0) * 255).astype(jnp.uint8)
    return self.State(videos=videos_uint8)
