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

"""NPZ Writer for Kauldron."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import io
import re
from typing import Any, Optional

from absl import logging
from etils import epath
from kauldron.train import auxiliaries
from kauldron.train import metric_writer
from kauldron.utils import chrono_utils
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member
import numpy as np
import optax


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class NpzWriter(metric_writer.KDMetricWriter):
  """KDMetricWriter that additionally saves array summaries to .npz files.

  Extends KDMetricWriter to dump a configurable subset of array-shaped
  summaries (and optionally scalars) to disk as NumPy `.npz` files, one file
  per logged step.

  Example usage:

  ```python
  cfg.writer = NpzWriter()
  ```

  Or with filtering:

  ```python
  cfg.writer = NpzWriter(
      key_patterns=["summaries/my_embedding", "summaries/logits.*"],
      save_scalars=True,
      output_dir=epath.Path("/path/to/array_dumps"),
  )
  ```
  """

  key_patterns: Sequence[str] | None = None
  save_scalars: bool = False
  output_dir: epath.Path | None = None

  def write_step_metrics(
      self,
      *,
      step: int,
      aux: auxiliaries.AuxiliariesState,
      schedules: Mapping[str, optax.Schedule],
      log_summaries: bool,
      timer: Optional[chrono_utils.Chrono] = None,
  ) -> None:
    super().write_step_metrics(
        step=step,
        aux=aux,
        schedules=schedules,
        log_summaries=log_summaries,
        timer=timer,
    )

    if not status.is_lead_host:
      return
    if not log_summaries:
      return

    aux_result = aux.compute(flatten=True)

    arrays_to_save = _filter_arrays(
        aux_result.summary_values, self.key_patterns
    )

    if self.save_scalars:
      scalars = aux_result.loss_values | aux_result.metric_values
      arrays_to_save |= _scalars_as_arrays(scalars)

    if arrays_to_save:
      self._save_npz(step, arrays_to_save)

  def _save_npz(self, step: int, arrays: dict[str, np.ndarray]) -> None:
    out_dir = self._get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{step:09d}.npz"
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    path.write_bytes(buf.getvalue())
    logging.info("NpzWriter: saved %d arrays to %s", len(arrays), path)

  def _get_output_dir(self) -> epath.Path:
    if self.output_dir is not None:
      return epath.Path(self.output_dir)
    self._assert_collection_is_set()
    return epath.Path(self.workdir) / "array_dumps" / self.collection


def _filter_arrays(
    values: dict[str, Any],
    patterns: Sequence[str] | None,
) -> dict[str, np.ndarray]:
  """Filters arrays based on key patterns."""
  compiled = [re.compile(p) for p in patterns] if patterns else None
  result = {}
  for key, value in values.items():
    if not isinstance(value, np.ndarray):
      continue
    if compiled is not None and not any(p.search(key) for p in compiled):
      continue
    result[key] = value
  return result


def _scalars_as_arrays(
    scalars: Mapping[str, Any],
) -> dict[str, np.ndarray]:
  return {k: np.asarray(v) for k, v in scalars.items()}
