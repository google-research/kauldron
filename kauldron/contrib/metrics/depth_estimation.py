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

"""Evaluation metrics for depth estimation."""

from __future__ import annotations

import dataclasses
from typing import Optional

import flax
import jax.numpy as jnp
from kauldron import kd
from kauldron import kontext
from kauldron.contrib.metrics import colormaps
from kauldron.metrics import base_state
from kauldron.typing import Bool, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MeanAbsoluteRelativeError(kd.metrics.Metric):
  """Mean of the absolute relative error, also known as REL or AbsRel."""

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  mask: Optional[kontext.Key] = None
  epsilon: float = 1e-6

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      preds: Float["*a"],
      targets: Float["*a"],
      mask: Optional[Bool["*a"] | Float["*a"]] = None,
  ) -> MeanAbsoluteRelativeError.State:

    # Compute absolute relative error.
    values = jnp.abs(preds - targets) / (targets + self.epsilon)

    # Averages across all dims.
    return self.State.from_values(values=values, mask=mask)


@typechecked
def _get_nearest_color_idx(
    tensor: Float["*b t h w k"], colormap: Float["n 3"]
) -> Float["*b t h w 1"]:
  """Given a tensor with RGB channels, returns the indices of nearest colors on the colormap."""
  colormap_expanded = jnp.expand_dims(colormap, axis=(0, 1, 2))

  distances = jnp.sum((tensor[..., None, :] - colormap_expanded) ** 2, axis=-1)

  # Find closest codebook vector indices
  nearest_color_idx = jnp.argmin(distances, axis=-1, keepdims=True).astype(
      tensor.dtype
  )
  return nearest_color_idx


def _invert_preprocess_depth(value, colormap, vmin=0.0, vmax=8.0):
  """Inverts preprocess_depth from dataset_utils.py."""

  # dequantize
  value = _get_nearest_color_idx(value, colormap).astype(jnp.float32) / 255.0
  value = ((vmax - vmin) * value) + vmin
  return value


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RGBMeanAbsoluteRelativeError(kd.metrics.Metric):
  """Depth performance metric (abs-rel) from RGB-space prediction."""

  pred: kontext.Key = kontext.REQUIRED
  target: kontext.Key = kontext.REQUIRED
  mask: Optional[kontext.Key] = None

  min_val: float = 1e-3
  max_val: float = 8.0
  epsilon: float = 1e-6

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      preds: Float["*b h w c"],
      targets: Float["*b h w c"],
      mask: Optional[Bool["*b h w 1"] | Float["*b h w 1"]] = None,
  ) -> RGBMeanAbsoluteRelativeError.State:

    # viridis colormap
    colormap = jnp.array(
        colormaps.viridis(),
        dtype=preds.dtype,
    )

    preds = _invert_preprocess_depth(preds, colormap)
    targets = _invert_preprocess_depth(targets, colormap)

    preds = jnp.clip(preds, self.min_val, self.max_val)
    targets = jnp.clip(targets, self.min_val, self.max_val)

    abs_rel = jnp.abs(preds - targets) / (targets + self.epsilon)

    return self.State.from_values(values=abs_rel, mask=mask)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class Delta1(kd.metrics.Metric):
  """Delta 1 metric for depth estimation.

  This metric calculates the percentage of pixels where the ratio of
  predicted to ground truth depth is within a threshold.
  """

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  mask: Optional[kontext.Key] = None
  threshold: float = 1.25
  epsilon: float = 1e-6
  denormalize: bool = False
  depth_mean: float = 0.0
  depth_std: float = 1.0

  @flax.struct.dataclass
  class State(base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      preds: Float["*a"],
      targets: Float["*a"],
      mask: Optional[Bool["*a"] | Float["*a"]] = None,
  ) -> Delta1.State:
    """Calculates the delta_1 metric.

    Args:
        preds: Predicted depth values.
        targets: Ground truth depth values.
        mask: Optional mask to apply to the values.

    Returns:
        The state of the metric.
    """
    if self.denormalize:
      preds = preds * self.depth_std + self.depth_mean
      targets = targets * self.depth_std + self.depth_mean

    # Clip the predictions to avoid negative values.
    preds = jnp.clip(preds, min=0.0)
    # Compute the ratio of predicted to ground truth depth.
    ratio = jnp.maximum(preds, targets) / (
        jnp.minimum(preds, targets) + self.epsilon
    )
    # Check if the ratio is within the threshold.
    values = ratio < self.threshold
    # Averages across all dims.
    return self.State.from_values(values=values.astype(jnp.float32), mask=mask)
