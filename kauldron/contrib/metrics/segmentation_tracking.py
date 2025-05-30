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

import dataclasses
import queue

import flax
import jax
import jax.numpy as jnp
from kauldron import kontext
from kauldron import metrics
from kauldron.typing import Float, typechecked  # pylint: disable=g-multiple-import,g-importing-member


def mask2heatmap(mask, h, w, height, width, max_num_labels=10):
  """Convert segmentation mask to one-hot heatmap using jax operations."""
  mask = jax.image.resize(mask, shape=(height, width), method="nearest")
  heatmap = jax.nn.one_hot(mask, num_classes=max_num_labels).astype(jnp.float32)
  heatmap = jax.image.resize(
      heatmap,
      shape=(h, w, max_num_labels),
      method="linear",
      antialias=False,
  )
  return heatmap


def label_propagation(
    feats,
    heatmap,
    n_context=20,
    temperature=0.7,
    topk=7,
    radius=20,
    restrict_neighborhood=True,
    norm_mask=False,
):
  """Propagation of the heatmap based on feature similarity."""
  h, w = feats.shape[1], feats.shape[2]

  # Creates a mask indicating valid neighbors for each grid element.
  gx, gy = jnp.meshgrid(
      jnp.arange(0, h), jnp.arange(0, w), indexing="ij"
  )  # (h, w)
  neighbor_mask = (gx[None, None, :, :] - gx[:, :, None, None]) ** 2 + (
      gy[None, None, :, :] - gy[:, :, None, None]
  ) ** 2
  neighbor_mask = neighbor_mask.astype(jnp.float32) ** 0.5
  neighbor_mask = (neighbor_mask < radius).astype(jnp.float32)  # (h, w, h, w)
  neighbor_mask = jnp.where(neighbor_mask == 0, -1e10, neighbor_mask)
  neighbor_mask = jnp.where(neighbor_mask == 1, 0, neighbor_mask)
  neighbor_mask = neighbor_mask.transpose(2, 3, 0, 1)  # (h, w, h, w)

  # The queue stores the context frames
  que = queue.Queue(n_context)
  for _ in range(n_context):
    que.put([feats[0], heatmap])

  preds = []
  for t in range(feats.shape[0]):
    # Use first and previous frames as context
    ctx_feats = jnp.stack([feats[0]] + [pair[0] for pair in que.queue])
    ctx_lbls = jnp.stack([heatmap] + [pair[1] for pair in que.queue])

    aff = (
        jnp.einsum("hwc, tmnc -> hwtmn", feats[t], ctx_feats) / temperature
    )  # (h, w, n_context+1, h, w)
    if restrict_neighborhood:
      aff.at[:, :, 1:].add(
          neighbor_mask[:, :, None]
      )  # (h, w, n_context+1, h, w)
    aff = aff.reshape(
        aff.shape[0], aff.shape[1], -1
    )  # (h, w, n_context+1 * h * w)

    weights, ids = jax.lax.top_k(aff, topk)  # (h, w, topk), (h, w, topk)
    weights = jax.nn.softmax(weights, axis=-1)  # (h, w, topk)
    ctx_lbls = ctx_lbls.reshape(
        -1, ctx_lbls.shape[-1]
    )  # (n_context+1 * h * w, n_class)
    pred = jnp.einsum(
        "hwlk, hwl -> hwk", ctx_lbls[ids], weights
    )  # (h, w, n_class)

    if que.qsize() == n_context:
      que.get()
    que.put([feats[t], pred])

    if norm_mask:
      pred -= pred.min(-1)[0][..., None]
      pred /= pred.max(-1)[0][..., None]

    preds.append(pred)
  preds = jnp.stack(preds)
  return preds


def heatmap2mask(preds, height, width, ori_height, ori_width):
  """Convert heatmap to segmentation mask (argmax and resize) using jnp."""

  def _process_frame(pred, height, width, ori_height, ori_width):
    pred = jax.image.resize(
        pred, (height, width, pred.shape[-1]), method="linear"
    )
    pred_lbl = jnp.argmax(pred, axis=-1)
    pred_lbl = jax.image.resize(
        pred_lbl, (ori_height, ori_width), method="nearest"
    )
    return pred_lbl

  pred_lbls = jax.vmap(_process_frame, in_axes=(0, None, None, None, None))(
      preds, height, width, ori_height, ori_width
  )

  return pred_lbls


def db_eval_iou(annotation, segmentation, void_pixels=None):
  """Compute region similarity as the Jaccard Index.

  Arguments:
    annotation   (ndarray): binary annotation map.
    segmentation (ndarray): binary segmentation map.
    void_pixels  (ndarray): optional mask with void pixels

  Returns:
    jaccard (float): region similarity
  """
  assert annotation.shape == segmentation.shape, (
      f"Annotation({annotation.shape}) and"
      f" segmentation:{segmentation.shape} dimensions do not match."
  )
  annotation = annotation.astype(bool)
  segmentation = segmentation.astype(bool)

  if void_pixels is not None:
    assert annotation.shape == void_pixels.shape, (
        f"Annotation({annotation.shape}) and void"
        f" pixels:{void_pixels.shape} dimensions do not match."
    )
    void_pixels = void_pixels.astype(bool)
  else:
    void_pixels = jnp.zeros_like(segmentation)

  # Intersection between all sets
  inters = jnp.sum(
      (segmentation & annotation) & jnp.logical_not(void_pixels), axis=(-2, -1)
  )
  union = jnp.sum(
      (segmentation | annotation) & jnp.logical_not(void_pixels), axis=(-2, -1)
  )

  j = inters / union
  if j.ndim == 0:
    j = 1 if jnp.isclose(union, 0) else j
  else:
    j = jnp.where(jnp.isclose(union, 0), 1, j)
  return j


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class SegmentationJaccard(metrics.Metric):
  """Computes IoU for video segmentation, first and last frame are omitted."""

  features: kontext.Key = kontext.REQUIRED  # e.g. "pred.features"
  gt_segmentations: kontext.Key = kontext.REQUIRED  # e.g. "batch.segmentations"
  patch_size: tuple[int, int] = (16, 16)
  n_max_class: int = 10
  resolution: tuple[int, int] = (480, 880)

  @flax.struct.dataclass
  class State(metrics.base_state.AverageState):
    pass

  @typechecked
  def get_state(
      self,
      features: Float["*b t k d"],
      gt_segmentations: Float["*b t H W"],
  ) -> SegmentationJaccard.State:
    """Computes the Jaccard metric between predicted and ground truth tracks."""
    height, width = self.resolution
    h, w = height // self.patch_size[0], width // self.patch_size[1]
    feats = features.reshape(features.shape[:-2] + (h, w, features.shape[-1]))
    feats = feats / jnp.linalg.norm(feats, axis=-1, keepdims=True)
    ori_height, ori_width = (
        gt_segmentations.shape[-2],
        gt_segmentations.shape[-1],
    )

    # Prepare downscaled first frame segmentation (resize and one-hot encode)
    feats = feats[0]  # Remove batch dimension, only works for batch_size=1
    gt_segmentations = gt_segmentations[0]  # Remove batch dimension
    lbls_small = mask2heatmap(
        gt_segmentations[0], h, w, height, width, self.n_max_class
    )
    pred_lbls = label_propagation(
        feats,
        lbls_small,
        n_context=20,
        temperature=0.7,
        topk=7,
        radius=20,
        restrict_neighborhood=True,
        norm_mask=False,
    )
    pred_lbls = heatmap2mask(pred_lbls, height, width, ori_height, ori_width)

    masks = gt_segmentations[1:-1]
    all_gt_masks = jax.nn.one_hot(masks, num_classes=self.n_max_class + 1)
    all_gt_masks = all_gt_masks[..., 1:]  # Remove background class

    masks = pred_lbls[1:-1]
    all_res_masks = jax.nn.one_hot(masks, num_classes=self.n_max_class + 1)
    all_res_masks = all_res_masks[..., 1:]  # Remove background class

    j_metrics_res = jax.vmap(db_eval_iou, in_axes=(-1, -1))(
        all_gt_masks, all_res_masks
    )

    j_metrics = jnp.nanmean(j_metrics_res, axis=-1)
    n_class = jnp.max(gt_segmentations)
    mask = (jnp.arange(self.n_max_class) < n_class).astype(jnp.float32)
    return self.State.from_values(values=j_metrics, mask=mask)
