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

"""Collection of losses commonly used for segmentation tasks."""

from __future__ import annotations

import dataclasses

import einops
import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron.losses import base
from kauldron.typing import Array, Dim, Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member
import optax


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class DiceLoss(base.Loss):
  """Compute the DICE loss, similar to IoU metric."""

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED

  @typechecked
  def get_values(
      self,
      preds: Float["*b h w m"],
      targets: Array["*b h w m"],
      num_objects: int = 1,
  ) -> Float["*b h w m"]:
    """Computes the DICE loss.

    Args:
        preds: The pre-sigmoid value for binary predictions for each example.
        targets: A float array with the same shape as inputs. Stores the binary
                  mask for each pixel in inputs.
        num_objects: Number of objects in the batch

    Returns:
        Dice loss value

    Based on:
      https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/training/loss_fns.py#L20C1-L49C36
    """

    preds = jax.nn.sigmoid(preds)
    numerator = 2 * jnp.sum(preds * targets, axis=(-2, -3), keepdims=True)
    denominator = jnp.sum(preds, axis=(-2, -3), keepdims=True) + jnp.sum(
        targets, axis=(-2, -3), keepdims=True
    )
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = einops.repeat(loss, "... 1 1 m -> ... h w m", h=Dim("h"), w=Dim("w"))

    return loss / num_objects


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class SigmoidFocalLoss(base.Loss):
  """Sigmoid focal loss used in RetinaNet (https://arxiv.org/abs/1708.02002)."""

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  alpha: float = 0.5
  gamma: float = 2.0

  @typechecked
  def get_values(
      self,
      preds: Float["*a"],
      targets: Array["*a"],
      num_objects: int = 1,
  ) -> Float["*a"]:
    """Computes the sigmoid focal loss.

    Args:
        preds: A float tensor, usually of shape [N, M, H, W]
                Mask predictions for each example.
        targets: A binary float tensor with the same shape as inputs.
        num_objects: Number of objects in the batch
    Returns:
        focal loss value

    Based on:
      https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/training/loss_fns.py#L52
    """

    prob = jax.nn.sigmoid(preds)
    ce_loss = optax.sigmoid_binary_cross_entropy(logits=preds, labels=targets)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** self.gamma)

    if self.alpha >= 0:
      alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
      loss = alpha_t * loss

    return loss / num_objects


@dataclasses.dataclass(eq=True, frozen=True, kw_only=True)
class IoULoss(base.Loss):
  """Simple IoU loss."""

  preds: kontext.Key = kontext.REQUIRED
  targets: kontext.Key = kontext.REQUIRED
  use_l1_loss: bool = True
  loss_on_multimask: bool = False

  @typechecked
  def get_values(
      self,
      preds: Float["*b m h w"],
      targets: Float["*b m h w"],
      pred_ious: Float["*b m"],
      num_objects: int,
  ) -> Float["*b m h w"]:
    """Gets the IoU loss.

    Args:
        preds: A float tensor of shape.
        targets: A float tensor with the same shape as inputs.
        pred_ious: A float tensor containing the predicted IoUs scores per mask
        num_objects: Number of objects in the batch

    Returns:
        IoU loss value

    Based on:
      https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/training/loss_fns.py#L93C1-L123C36
    """

    assert preds.ndim == 4 and targets.ndim == 4
    pred_mask = einops.rearrange(preds, "... h w -> ... (h w)") > 0
    gt_mask = einops.rearrange(targets, "... h w -> ... (h w)") > 0
    area_i = jnp.sum(pred_mask & gt_mask, axis=-1).astype(jnp.float32)
    area_u = jnp.sum(pred_mask | gt_mask, axis=-1).astype(jnp.float32)
    actual_ious = area_i / jnp.clip(area_u, a_min=1.0)

    if self.use_l1_loss:
      loss = jnp.abs(pred_ious - actual_ious)
    else:
      loss = (pred_ious - actual_ious)**2

    return loss / num_objects
