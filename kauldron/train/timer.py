# Copyright 2024 The kauldron Authors.
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

"""Timer for measuring training performance."""

from __future__ import annotations

import contextlib
import time

from kauldron import checkpoints
from orbax import checkpoint as ocp


class PerformanceTimer(checkpoints.items.CheckpointItem):
  """Timer for measuring training performance."""

  def __init__(
      self,
      # how many iters the model has already been trained for
      initial_step_num,
      # how long it took to train the model so far
      initial_training_time_hours,
      per_device_batch_size,
      global_batch_size,
  ):
    self.step_num_when_last_logged = initial_step_num
    self.per_device_batch_size = per_device_batch_size
    self.global_batch_size = global_batch_size

    # We assume that training (re)starts NOW.
    self.initial_training_time_hours = initial_training_time_hours
    self.time_when_finished_last_step = time.monotonic()
    self.time_when_last_logged = self.time_when_finished_last_step
    # offest by the time already elapsed
    self.time_at_the_start_of_training = (
        self.time_when_finished_last_step - initial_training_time_hours * 3600.0
    )

    # Records any time that is not included in steps_per_sec computation.
    self.skip_time_steps = 0.0
    # Records any time that is not included in total_training_time.
    self.skip_time_total = 0.0

  def finish_step(self):
    self.time_when_finished_last_step = time.monotonic()

  @property
  def total_training_time_hours(self):
    return (
        self.time_when_finished_last_step - self.time_at_the_start_of_training
    ) / 3600.0

  def log_stats(self, step_num):
    """Computes performance stats."""
    num_steps_sice_logged = step_num - self.step_num_when_last_logged
    elapsed_time_since_logged = (
        self.time_when_finished_last_step - self.time_when_last_logged
    )

    # Subtract any time that was skipped and reset the skipped time counter.
    elapsed_time_since_logged -= self.skip_time_steps
    self.skip_time_steps = 0.0

    steps_per_sec = num_steps_sice_logged / elapsed_time_since_logged
    self.step_num_when_last_logged = step_num
    self.time_when_last_logged = self.time_when_finished_last_step

    return dict(
        steps_per_sec=steps_per_sec,
        data_points_per_sec_per_device=(
            steps_per_sec * self.per_device_batch_size
        ),
        data_points_per_sec_global=steps_per_sec * self.global_batch_size,
        total_training_time_hours=self.total_training_time_hours,
    )

  @contextlib.contextmanager
  def exclude_from_step_stats(self):
    section_start_time = time.monotonic()
    try:
      yield
    finally:
      section_time = time.monotonic() - section_start_time
      self.skip_time_steps += section_time

  @contextlib.contextmanager
  def exclude_from_total_time(self):
    section_start_time = time.monotonic()
    try:
      yield
    finally:
      section_time = time.monotonic() - section_start_time
      self.skip_time_total += section_time

  # Handle orbax checkpointing

  def __kd_ocp_handlers__(self) -> ocp.CheckpointHandler:
    return ocp.JsonCheckpointHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    return ocp.args.JsonSave({
        'training_time_hours': self.total_training_time_hours,
    })

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    return ocp.args.JsonRestore()

  def __kd_ocp_restore_post__(self, value) -> PerformanceTimer:
    return type(self)(
        initial_step_num=self.step_num_when_last_logged,
        initial_training_time_hours=value['training_time_hours'],
        per_device_batch_size=self.per_device_batch_size,
        global_batch_size=self.global_batch_size,
    )
