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

"""ImageGrid Summary."""

from __future__ import annotations

import dataclasses
from typing import Self, Sequence, TypeAlias, cast

import flax
import jax
from kauldron import kd
from kauldron.typing import Float, Float32, Shape, UInt8, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import numpy as np
import PIL
import PIL.ImageColor
import PIL.ImageDraw
import PIL.ImageFont


ColorLike: TypeAlias = (
    str | tuple[int, int, int] | tuple[float, float, float] | float
)


@dataclasses.dataclass(kw_only=True, frozen=True)
class ImageGrid(kd.metrics.Metric):
  """Show a grid of images as a single image with borders and column headers.

  Example Usage:
  ```python
  cfg.train_summaries["image_grid"] = ImageGrid(
      columns={
          "gt": kd.summaries.ShowImages(images="batch.image"),
          "pred": kd.summaries.ShowImages(images="preds.image"),
      }
  )
  ```
  Or equivalently:
  ```python
  cfg.train_summaries["image_grid"] = ImageGrid.simple(
      columns={
          "gt": "batch.image",
          "pred": "preds.image",
      }
  )
  ```
  Which would log a single image like this:

  |------------|------------|
  |     gt     |    pred    |
  |------------|------------|
  |            |            |
  |   [IMG1]   |   [IMG1]   |
  |            |            |
  | -----------|------------|
  |            |            |
  |   [IMG2]   |   [IMG2]   |
  |            |            |
  |------------|------------|
  | ...        | ...        |


  Attributes:
    columns: A dictionary of column names to metrics that compute the images for
      each column, for example like `kd.summaries.ShowImages` or
      `kd.summaries.ShowSegmentations`.
    border_width: The width of the border around each image in pixels.
    bg_color: The background color of the image. Can be a name like "black", a
      hex color like "#000000", an int tuple (r, g, b) with values in [0, 255],
      a float tuple (r, g, b) with values in [0, 1], or a single float in [0, 1]
      (indicating a fraction of white).
    font_size: The font size of the column names.
    font_color: The font color of the column names. Same format as `bg_color`.
  """

  columns: dict[str, kd.metrics.Metric]
  border_width: int = 2
  bg_color: ColorLike = "black"
  font_size: int = 10
  font_color: ColorLike = "white"

  @flax.struct.dataclass
  class State(kd.metrics.State["ImageGrid"]):
    """Collect the states of the per-column submetrics."""

    show_image_states: Sequence[kd.metrics.State]

    @classmethod
    def empty(cls) -> Self:
      return cls(show_image_states=tuple())

    def merge(self, other) -> Self:
      if not self.show_image_states:
        return other

      substates_merged = []
      for i, substate in enumerate(self.show_image_states):
        substates_merged.append(substate.merge(other.show_image_states[i]))

      return dataclasses.replace(
          self, show_image_states=tuple(substates_merged)
      )

    def finalize(self) -> Self:
      return dataclasses.replace(
          self,
          show_image_states=tuple(s.finalize() for s in self.show_image_states),
      )

    @typechecked
    def compute(self) -> Float["1 H W 3"]:
      """Computes final metrics from intermediate values."""
      images = {}
      for colname, substate in zip(self.parent.columns, self.show_image_states):
        images[colname] = _normalize_image(substate.compute())

      bg_color = _normalize_color(self.parent.bg_color)
      font_color = _normalize_color(self.parent.font_color)

      return _render_image_grid(
          images,
          font_size=self.parent.font_size,
          border_width=self.parent.border_width,
          bg_color=bg_color,
          font_color=font_color,
      )

  def get_state(self, **kwargs) -> State:
    """Returns the state of the metric."""
    substates = []
    for colname, col_summary in self.columns.items():
      relevant_kwargs = {
          k.removeprefix(f"{colname}_"): v
          for k, v in kwargs.items()
          if k.startswith(f"{colname}_")
      }
      substates.append(col_summary.get_state(**relevant_kwargs))
    return self.State(show_image_states=tuple(substates))

  @classmethod
  def simple(
      cls,
      columns: dict[str, str],
      num_images: int = 5,
      in_vrange: tuple[float, float] | None = None,
      **kwargs,
  ):
    """Returns a simple ImageGrid of ShowImages summaries.

    Args:
      columns: A dictionary of column names to key paths for ShowImages.
      num_images: The number of images to show in each column.
      in_vrange: The range to normalize the images to.
      **kwargs: Additional arguments to pass to ImageGrid.

    Returns:
      An ImageGrid summary.
    """

    def _make_show_images(keypath: str):
      return kd.summaries.ShowImages(
          images=keypath, num_images=num_images, in_vrange=in_vrange
      )

    return cls(
        columns=jax.tree.map(_make_show_images, columns),
        **kwargs,
    )

  def __kontext_keys__(self) -> dict[str, kd.kontext.Key]:
    keypaths = {}
    for colname, col_summary in self.columns.items():
      keypaths.update({
          f"{colname}_{k}": v
          for k, v in kd.kontext.get_keypaths(col_summary).items()
      })
    return keypaths


def _normalize_color(color: ColorLike) -> tuple[int, int, int]:
  match color:
    case str():
      rgb = PIL.ImageColor.getcolor(color, mode="RGB")
      return cast(tuple[int, int, int], rgb)
    case float():
      if 0 <= color <= 1:
        raise ValueError(
            "When passing color as a single float, its value must be in "
            f"[0, 1], but got {color=}."
        )
      return (int(color * 255), int(color * 255), int(color * 255))
    case (int(r), int(g), int(b)):
      if not all(0 <= c <= 255 for c in (r, g, b)):
        raise ValueError(
            "When passing color as an int tuple, its values must be in "
            f"[0, 255], but got {color=}."
        )
      return (r, g, b)
    case (float(r), float(g), float(b)):
      if not all(0.0 <= c <= 1.0 for c in (r, g, b)):
        raise ValueError(
            "When passing color as a float tuple, its values must be in "
            f"[0, 1], but got {color=}."
        )
      return tuple(int(c * 255) for c in (r, g, b))
    case _:
      raise ValueError(f"Unsupported color type: color ({type(color)})={color}")


@typechecked
def _normalize_image(
    image: Float["*b h w #3"] | UInt8["*b h w #3"],
) -> UInt8["*b h w #3"]:
  if isinstance(image, UInt8["*b h w #3"]):
    return image
  elif isinstance(image, Float["*b h w #3"]):
    return (image * 255.0).clip(0, 255).astype(np.uint8)
  else:
    raise ValueError(f"Unsupported image type: image ({type(image)})={image}")


@typechecked
def _render_image_grid(
    images: dict[str, UInt8["n h w #3"]],
    font_size: int,
    border_width: int,
    bg_color: tuple[int, int, int],
    font_color: tuple[int, int, int],
) -> Float32["1 H W 3"]:
  """Renders a grid of images with column headers."""
  font = PIL.ImageFont.load_default(font_size)
  n, h, w = Shape("n h w")
  m = len(images)
  text_height = font.getbbox("".join(images.keys()))[3]
  b = border_width

  total_height = (n + 2) * b + n * h + text_height
  total_width = (m + 1) * b + m * w

  # Create a blank image with the correct size and background color
  img = PIL.Image.new("RGB", (total_width, total_height), bg_color)
  draw = PIL.ImageDraw.Draw(img)
  # Render column names
  for i, col_name in enumerate(images):
    column_offset = b + i * (w + b)
    col_name_width = font.getlength(col_name)
    centered_text_offset = round((w - col_name_width) / 2)
    draw.text(
        (column_offset + centered_text_offset, b),
        col_name,
        fill=font_color,
        font=font,
    )

  img_arr = np.array(img)
  for col, col_images in enumerate(images.values()):
    for row in range(n):
      x = b + col * (w + b)
      y = 2 * b + text_height + row * (h + b)
      img_arr[y : y + h, x : x + w, :] = col_images[row]

  return img_arr[None].astype(np.float32) / 255.0
