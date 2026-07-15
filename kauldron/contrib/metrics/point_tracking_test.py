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

from kauldron.contrib.metrics import point_tracking
import numpy as np


def test_tap_occlusion_accuracy_padded_batch_deflation():
  # B=2, T=3, Q=1, C=1 (axis_order="BTQC")
  # Batch 0: Valid
  # Batch 1: Padded (Fully Invalid)

  pred_visible = np.array(
      [
          [[[1]], [[0]], [[1]]],  # Batch 0
          [[[0]], [[0]], [[0]]],  # Batch 1 (Padded)
      ],
      dtype=np.float32,
  )

  gt_visible = np.array(
      [
          [[[1]], [[1]], [[1]]],  # Batch 0
          [[[1]], [[1]], [[1]]],  # Batch 1 (Padded, garbage GT)
      ],
      dtype=np.float32,
  )

  evaluation_mask = np.array(
      [
          [[1], [1], [0]],  # Batch 0 (frame 2 is invalid)
          [[0], [0], [0]],  # Batch 1 (All invalid)
      ],
      dtype=np.float32,
  )

  metric = point_tracking.TapOcclusionAccuracy(axis_order="BTQC")
  state = metric.get_state(
      pred_visible=pred_visible,
      gt_visible=gt_visible,
      evaluation_mask=evaluation_mask,
  )

  # Batch 0: frames 0 and 1 are valid.
  # Frame 0: Pred 1, GT 1 -> Correct
  # Frame 1: Pred 0, GT 1 -> Incorrect
  # Accuracy = 0.5

  # Batch 1: All invalid.

  # Old code would compute accuracy = 0 for Batch 1.
  # And then State.compute() would average Batch 0 (0.5) and
  # Batch 1 (0.0) -> 0.25 (Deflation).

  # New code should ignore Batch 1 and give 0.5.

  # We assert 0.5 here to test NEW code. It should FAIL on old code.
  result = state.compute()
  np.testing.assert_allclose(result, 0.5)


def test_tap_occlusion_accuracy_padded_tracks_inflation():
  # B=1, T=3, Q=2, C=1 (axis_order="BTQC")
  # Track 0: Valid, 1/2 correct (excluding query at T=0)
  # Track 1: Padded, Trivial match (0==0)

  pred_visible = np.array(
      [[[[1], [0]], [[0], [0]], [[1], [0]]]],  # T=0 (Query)  # T=1  # T=2
      dtype=np.float32,
  )

  gt_visible = np.array(
      [[[[1], [0]], [[1], [0]], [[1], [0]]]],  # T=0 (Query)  # T=1  # T=2
      dtype=np.float32,
  )

  metric = point_tracking.TapOcclusionAccuracy(axis_order="BTQC")

  # 1. Leaky Heuristic (No mask)
  state_leaky = metric.get_state(
      pred_visible=pred_visible,
      gt_visible=gt_visible,
      # evaluation_mask=None
  )
  # Heuristic assumes Query is T=0.
  # Evaluates T=1, T=2.
  # Track 0: T=1 wrong, T=2 correct -> 1/2 correct.
  # Track 1: T=1 correct (0==0), T=2 correct (0==0) -> 2/2 correct.
  # Global: 3/4 correct = 0.75.

  # 2. Correct Masking
  evaluation_mask = np.array(
      [[[1], [0], [1], [0], [0], [0]]], dtype=np.float32  # T=0  # T=1  # T=2
  ).reshape(1, 3, 2)

  state_correct = metric.get_state(
      pred_visible=pred_visible,
      gt_visible=gt_visible,
      evaluation_mask=evaluation_mask,
  )
  # Mask evaluates Track 0 at T=0, T=1.
  # Track 0: T=0 correct, T=1 wrong -> 1/2 correct.
  # Track 1 is ignored.
  # Global: 1/2 correct = 0.5.

  print(f"DEBUG: Leaky result = {state_leaky.compute()}")
  print(f"DEBUG: Correct result = {state_correct.compute()}")

  # Verify the bug: Heuristic is inflated (0.75) compared to correct (0.5)
  np.testing.assert_allclose(state_leaky.compute(), 0.75)
  np.testing.assert_allclose(state_correct.compute(), 0.5)


def test_tap_position_accuracy_query_frame_leakage():
  # B=1, T=3, Q=1, C=2 (BTQC)
  # Query frame is T=1.
  # Model is initialized at T=1, so pred matches GT perfectly there.
  # Model is wrong at T=0 and T=2.

  pred_tracks = np.array(
      [[
          [[0.0, 0.0]],  # T=0 (Wrong)
          [[0.5, 0.5]],  # T=1 (Query - Perfect)
          [[0.0, 0.0]],  # T=2 (Wrong)
      ]],
      dtype=np.float32,
  )

  gt_tracks = np.array(
      [[[[0.5, 0.5]], [[0.5, 0.5]], [[0.5, 0.5]]]],  # T=0  # T=1  # T=2
      dtype=np.float32,
  )

  gt_visible = np.array(
      [[[[1]], [[1]], [[1]]]], dtype=np.float32  # T=0  # T=1  # T=2
  )

  metric = point_tracking.TapPositionAccuracy(
      axis_order="BTQC", thresholds=(0.1,)
  )

  # 1. Leaky Heuristic (No mask, no query_frame passed -> defaults to 0)
  state_leaky = metric.get_state(
      pred_tracks=pred_tracks,
      gt_tracks=gt_tracks,
      gt_visible=gt_visible,
  )
  # Heuristic assumes Query is T=0.
  # Evaluates T=1, T=2.
  # T=1 is perfect -> counts as correct!
  # T=2 is wrong.
  # Score = 0.5 (1/2 correct).

  # 2. Correct Masking (Excludes Query T=1)
  evaluation_mask = np.array(
      [[
          [1],  # T=0 (Evaluate)
          [0],  # T=1 (Query - Ignore)
          [1],  # T=2 (Evaluate)
      ]],
      dtype=np.float32,
  ).reshape(1, 3, 1)

  state_correct = metric.get_state(
      pred_tracks=pred_tracks,
      gt_tracks=gt_tracks,
      gt_visible=gt_visible,
      evaluation_mask=evaluation_mask,
  )
  # Evaluates T=0, T=2. Both wrong.
  # Score = 0.0.

  print(f"DEBUG: Leaky result = {state_leaky.compute()}")
  print(f"DEBUG: Correct result = {state_correct.compute()}")

  # Verify the bug: Heuristic is inflated (0.5) compared to correct (0.0)
  np.testing.assert_allclose(state_leaky.compute(), 0.5)
  np.testing.assert_allclose(state_correct.compute(), 0.0)
