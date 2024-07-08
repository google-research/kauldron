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

"""Timer class."""

from __future__ import annotations

import collections
from collections.abc import Iterator
import contextlib
import dataclasses
import enum
import functools
import time
from typing import Self

import jax
from kauldron import checkpoints
from kauldron.utils import kdash
from orbax import checkpoint as ocp


# TODO(epot): Could add a mode to report individual one-off events
# (compilation time,...) ?


class Pause(enum.StrEnum):
  """Pause names.

  Only exists so the `chrono.pause()` calls match the reported metrics.
  Ideally, user should call `chrono.pause('checkpoint')` directly and the
  dashboard would be automatically updated to add the matching plot.
  Currently, the dashboard names are hardcoded through this enum.
  """

  CHECKPOINT = 'checkpoint'
  EVALS_ALONG_TRAIN = 'evals_along_train'
  METRICS_WRITING = 'metrics_writing'


@dataclasses.dataclass(kw_only=True)
class _TimerElement:
  """Track time for an individual metric."""

  num_steps: int = 0
  total_time: float = 0.0

  @property
  def steps_per_sec(self) -> float:
    return self.num_steps / self.total_time

  @property
  def secs_per_step(self) -> float:
    return self.total_time / self.num_steps

  def reset(self) -> None:
    self.num_steps = 0
    self.total_time = 0.0


@dataclasses.dataclass(kw_only=True)
class Chrono(checkpoints.items.CheckpointItem):
  """Timer class to record various durations.

  Usage:

  ```python
  chrono = kd.utils.chrono.Chrono(name='name')

  chrono.start_loop()
  for ex in ds:
    ...
    chrono.finish_step()
  ```

  Attributes:
    name: Name of the chrono / dahsboard collection.
    batch_size: Global batch size. If set, will report the number of examples
      per sec.
    pause_names: Name of additional metrics to plot on the dashboard. Should
      match the `chrono.pause(report_as='name')` call.
    chrono_since_last_flush: Individual timers.
  """

  name: str
  batch_size: int | None = None
  pause_names: list[str] = dataclasses.field(default_factory=list)

  chrono_since_last_flush: dict[str, _TimerElement] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(_TimerElement)
  )

  # Time when the current step was started.
  _start_time: float = 0.0
  # Total global time (including all pause time)
  _global_total_time: float = 0.0

  def start_loop(self) -> None:
    """Starts the chrono.

    Should be called only once before starting a loop.

    We do not use context manager here to ensure the iteration
    (`for ex in ds:`) is also accounted for.
    """
    self._start_time = time.perf_counter()
    # `.start_loop()` can be called multiple times.
    # * For every `.evaluate()`: Here `.flush_metrics()` should have been called
    #   previously, so metrics should be empty already.
    # * After restauring from a checkpoint (is a no-op for the current step)

  def finish_step(self) -> None:
    """Notify the end of a step."""
    end_time = time.perf_counter()
    elapsed_time = end_time - self._start_time
    self._start_time = end_time  # Reset the start time for the next step.

    # End of the step, update the chrono.
    chrono = self.chrono_since_last_flush[self.name]
    chrono.num_steps += 1
    chrono.total_time += elapsed_time

    self._global_total_time += elapsed_time

  @contextlib.contextmanager
  def pause(self, report_as: str | None = None) -> Iterator[None]:
    """Pause the main chrono and start a sub-chrono.

    Pause times cannot be nested.

    Args:
      report_as: If set, the pause time will be reported in a separate plot.

    Yields:
      None
    """
    start = time.perf_counter()
    try:
      yield
    finally:
      end = time.perf_counter()
      elapsed = end - start

      self._start_time += elapsed  # Add the pause time to the main chrono.
      self._global_total_time += elapsed  # Pause time is included in the total

      if report_as:  # Track pause time in a separate chrono.
        # TODO(epot): Raise error if `report_as` do not match self.pause_names
        chrono = self.chrono_since_last_flush[f'{report_as}']
        chrono.num_steps += 1
        chrono.total_time += elapsed

  def flush_metrics(self) -> dict[str, float]:
    """Returns the metrics and resets the chronos."""
    # Report the main trainer chrono
    chrono = self.chrono_since_last_flush[self.name]
    metrics = {
        'steps_per_sec': chrono.steps_per_sec,
        'total_training_time_hours': self._global_total_time / (60.0 * 60.0),
    }
    if self.batch_size:
      data_points_per_sec_global = (
          chrono.num_steps * self.batch_size
      ) / chrono.total_time
      metrics.update({
          'data_points_per_sec_global': data_points_per_sec_global,
          # The ex per sec per device allow to get a normalized performance
          # metric, independent of the batch size or platform.
          'data_points_per_sec_per_device': (
              data_points_per_sec_global / jax.device_count()
          ),
      })

    # Report the extra metrics (checkpoint) and reset the chrono
    for name, chrono in self.chrono_since_last_flush.items():
      if chrono.num_steps:
        metrics[f'{name}/avg_time_sec'] = chrono.secs_per_step
      chrono.reset()
    return metrics

  # ========== Dashboard protocol ==========

  @functools.cached_property
  def __dashboards__(self) -> kdash.DashboardsBase:
    # Default plot
    # TODO(epot): Better plots
    # * `steps_per_sec` should report both with and without overhead
    # * Merge `data_points_per_sec_global` / `data_points_per_sec_per_device`
    #   in a single plot.

    keys = [  # pylint: disable=invalid-name
        'steps_per_sec',
        'total_training_time_hours',
    ]
    if self.batch_size:
      keys.extend([
          'data_points_per_sec_global',
          'data_points_per_sec_per_device',
      ])
    plots = [
        kdash.Plot(
            y_key=f'perf_stats/{key}',
            collections=[self.name],
        )
        for key in keys
    ]

    # Collect all the average times.
    plots.append(
        kdash.Plot(
            y_key='perf_stats/avg_time_sec',
            collection_to_ykeys={
                self.name: [
                    f'perf_stats/{key}/avg_time_sec'
                    for key in [self.name] + self.pause_names
                ]
            },
        ),
    )

    # TODO(epot): Add plot of steps per timestamp (to check prehemptions,...)
    # * Customize title
    # * Should instead be the relative time since the start (e.g. in hours/days)
    # plots.append(
    #     kdash.Plot(
    #         y_key='step',
    #         x_key='timestamp',
    #         collections=[self.name],
    #     ),
    # )

    return kdash.SingleDashboard(
        name='perf_stats',
        title='{xid}: Performance statistics',
        plots=plots,
    )

  # ========== Checkpoint protocol ==========

  def __kd_ocp_handlers__(self) -> ocp.CheckpointHandler:
    return ocp.JsonCheckpointHandler()

  def __kd_ocp_save_args__(self) -> ocp.args.CheckpointArgs:
    data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    data['pause_names'] = [str(p) for p in self.pause_names]
    data['chrono_since_last_flush'] = {
        k: dataclasses.asdict(v)
        for k, v in self.chrono_since_last_flush.items()
    }
    return ocp.args.JsonSave(data)

  def __kd_ocp_restore_args__(self) -> ocp.args.CheckpointArgs:
    return ocp.args.JsonRestore()

  def __kd_ocp_restore_post__(self, value) -> Self:
    value['chrono_since_last_flush'] = collections.defaultdict(
        _TimerElement,
        {
            k: _TimerElement(**kwargs)
            for k, kwargs in value['chrono_since_last_flush'].items()
        },
    )
    return type(self)(**value)
