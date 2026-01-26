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

"""GIF Video Writer for Kauldron.

A custom KDMetricWriter that encodes video summaries as animated GIFs and
writes them to TensorBoard.

Example usage:
  cfg.writer = GifVideoWriter()
  cfg.train_summaries["my_video"] = ShowVideosAsGif(
      videos="batch.video",
      in_vrange=(-1.0, 1.0),
  )
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import io
import threading
from typing import Optional

import jax
from kauldron.train import auxiliaries
from kauldron.train import metric_writer
from kauldron.typing import UInt8
from kauldron.utils import chrono_utils
import numpy as np
import optax
import PIL.Image as pil_image

import tensorboard.compat
from tensorboard.compat.proto import summary_pb2

tf = tensorboard.compat.tf2


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class GifVideoWriter(metric_writer.KDMetricWriter):
  """Custom MetricWriter that encodes video summaries as animated GIFs.

  Extends KDMetricWriter to filter UInt8 5D video arrays from summary values
  and encode them as animated GIF/WEBP/PNG images that display in TensorBoard.

  This writer looks for summary values that are UInt8 5D arrays with shape
  [N, T, H, W, C] (batch, time, height, width, channels) and encodes them as
  animated images.

  To control the output format per-summary, add a `save_format` attribute to
  the metric (e.g., 'GIF', 'WEBP', or 'PNG'). Default is 'GIF'.

  Example:
    cfg.writer = GifVideoWriter()
  """

  write_video_summaries: bool = True

  def write_animation(
      self,
      step: int,
      videos: Mapping[str, UInt8["n t h w #3"]],
      img_format: str,
  ):
    """Write animated image summaries to TensorBoard."""
    if hasattr(self._tf_summary_writer, "_summary_writer"):
      with self._tf_summary_writer._summary_writer.as_default():  # pylint: disable=protected-access
        for key, value in videos.items():
          _animated_image(
              key,
              value,
              step=step,
              max_outputs=value.shape[0],
              img_format=img_format,
          )

  def write_step_metrics(
      self,
      *,
      step: int,
      aux: auxiliaries.AuxiliariesState,
      schedules: Mapping[str, optax.Schedule],
      log_summaries: bool,
      timer: Optional[chrono_utils.Chrono] = None,
  ) -> None:
    """Write step metrics including video summaries as animated GIFs."""
    super().write_step_metrics(
        step=step,
        aux=aux,
        schedules=schedules,
        log_summaries=log_summaries,
        timer=timer,
    )

    if self.write_video_summaries and log_summaries:
      aux_result = aux.compute(flatten=True)
      with jax.transfer_guard("allow"):
        video_summaries = {
            name: np.asarray(value)
            for name, value in aux_result.summary_values.items()
            if isinstance(value, UInt8["n t h w #3"])
        }

        if not video_summaries:
          return

        save_format_dict = {"PNG": [], "GIF": [], "WEBP": []}
        for name in video_summaries:
          base_name = name.replace("summaries/", "")
          if aux.summary_states and base_name in aux.summary_states:
            parent = aux.summary_states[base_name].parent
            if hasattr(parent, "save_format"):
              fmt = parent.save_format
              if fmt in save_format_dict:
                save_format_dict[fmt].append(name)
                continue
          save_format_dict["GIF"].append(name)

        for name, video in video_summaries.items():
          if video.size == 0:
            raise ValueError(
                f"Video summary `{name}` is empty array of shape {video.shape}."
            )

        for save_format, names in save_format_dict.items():
          if names:
            videos_for_format = {k: video_summaries[k] for k in names}
            self.write_animation(
                step=step,
                videos=videos_for_format,
                img_format=save_format,
            )


def _animated_image(
    name, data, step=None, max_outputs=3, description=None, img_format="GIF"
):
  """Write animated image summary to TensorBoard."""

  def encode(video_array: np.ndarray, **kwargs) -> bytes:
    frames = [pil_image.fromarray(frame) for frame in video_array]
    output = io.BytesIO()
    if img_format == "WEBP":
      frames[0].save(
          output,
          format=img_format,
          save_all=True,
          append_images=frames[1:],
          duration=100,
          loop=0,
          quality=80,
      )
    elif img_format == "GIF":
      palette = frames[0].quantize(
          method=pil_image.MEDIANCUT, dither=pil_image.Dither.NONE
      )
      frames = [
          f.quantize(palette=palette, dither=pil_image.Dither.NONE)
          for f in frames
      ]
      frames[0].save(
          output,
          format=img_format,
          loop=0,
          save_all=True,
          append_images=frames[1:],
          optimize=True,
          disposal=2,
          **kwargs,
      )
    else:
      frames[0].save(
          output,
          format=img_format,
          loop=0,
          save_all=True,
          append_images=frames[1:],
          **kwargs,
      )
    return output.getvalue()

  summary_metadata = summary_pb2.SummaryMetadata(
      display_name=None,
      summary_description=description,
      plugin_data=summary_pb2.SummaryMetadata.PluginData(
          plugin_name="images",
          content=b"",
      ),
  )
  summary_scope = (
      getattr(tf.summary.experimental, "summary_scope", None)
      or tf.summary.summary_scope
  )
  with summary_scope(
      name, "image_summary", values=[data, max_outputs, step]
  ) as (tag, _):

    @_LazyTensorCreator
    def lazy_tensor():
      tf.debugging.assert_rank(data, 5)
      tf.debugging.assert_non_negative(max_outputs)
      encoded_vids = tf.convert_to_tensor(
          [encode(vid) for vid in data[:max_outputs]]
      )
      dimensions = tf.stack(
          [
              tf.as_string(data.shape[2], name="width"),
              tf.as_string(data.shape[1], name="height"),
          ],
          name="dimensions",
      )
      return tf.concat([dimensions, encoded_vids], axis=0)

    return tf.summary.write(
        tag=tag, tensor=lazy_tensor, step=step, metadata=summary_metadata
    )


_CALL_IN_PROGRESS_SENTINEL = object()


class _LazyTensorCreator:
  """Lazy wrapper for callable that returns tf.Tensor, auto-converted."""

  def __init__(self, tensor_callable):
    if not callable(tensor_callable):
      raise ValueError("Not a callable: %r" % tensor_callable)
    self._tensor_callable = tensor_callable
    self._tensor = None
    self._tensor_lock = threading.RLock()
    _register_conversion_function_once()

  def __call__(self):
    if self._tensor is None or self._tensor is _CALL_IN_PROGRESS_SENTINEL:
      with self._tensor_lock:
        if self._tensor is _CALL_IN_PROGRESS_SENTINEL:
          raise RuntimeError(
              "Cannot use LazyTensorCreator with reentrant callable"
          )
        elif self._tensor is None:
          self._tensor = _CALL_IN_PROGRESS_SENTINEL
          self._tensor = self._tensor_callable()
    return self._tensor


def _lazy_tensor_creator_converter(value, dtype=None, name=None, as_ref=False):
  """Convert _LazyTensorCreator to a Tensor for TensorFlow operations."""

  del name
  if not isinstance(value, _LazyTensorCreator):
    raise RuntimeError("Expected LazyTensorCreator, got %r" % value)
  if as_ref:
    raise RuntimeError("Cannot use LazyTensorCreator to create ref tensor")
  tensor = value()
  if dtype not in (None, tensor.dtype):
    raise RuntimeError(
        "Cannot convert LazyTensorCreator returning dtype %s to dtype %s"
        % (tensor.dtype, dtype)
    )
  return tensor


_conversion_registered = False
_conversion_registered_lock = threading.Lock()


def _register_conversion_function_once():
  global _conversion_registered
  if not _conversion_registered:
    with _conversion_registered_lock:
      if not _conversion_registered:
        _conversion_registered = True
        tf.register_tensor_conversion_function(
            base_type=_LazyTensorCreator,
            conversion_func=_lazy_tensor_creator_converter,
            priority=0,
        )
