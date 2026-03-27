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

"""Helper class for mutating configs for testing purposes."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

from etils import epath
from etils import epy
import jax
from kauldron import konfig
from kauldron.cli import cmd_utils as cu

# pylint: disable=g-import-not-at-top
with konfig.imports(lazy=True):
  from kauldron import evals as kd_evals
  from kauldron import inspect as kd_inspect
  from jax.experimental import checkify as jax_checkify
# pylint: enable=g-import-not-at-top

BATCH_SIZE_DEVICES = "devices"


@dataclasses.dataclass(frozen=True, kw_only=True)
class PatchConfig:
  """Patches a config, in order to speed up test runs."""

  stop_after_steps: int | None = 1
  """Stop training after N steps. Defaults to 1."""

  batch_size: int | str | None = BATCH_SIZE_DEVICES
  """Override the batch size in train and eval datasets."""

  num_batches: int | None = 1
  """Override the number of batches in train and eval datasets."""

  skip_eval: bool = False
  """Whether to skip eval."""

  eval_run_once: bool = True
  """Run each evaluator only once (at step 0)."""

  skip_checkpointer: bool = True
  """Whether to skip checkpointer."""

  skip_metrics: bool = False
  """Whether to skip computing train metrics."""

  skip_summaries: bool = False
  """Whether to skip computing train summaries."""

  skip_flatboard: bool = True
  """Whether to skip flatboard dashboard creation."""

  profile: bool = False
  """Whether to enable profiling."""

  checkify: bool = False
  """Whether to enable jax.experimental.checkify."""

  def __call__(
      self, cfg: konfig.ConfigDict
  ) -> tuple[konfig.ConfigDict, dict[str, Any]]:
    """Returns a patched copy of the config along with a dict of updates."""
    cfg = copy.deepcopy(cfg)

    updates = {}
    if self.stop_after_steps is not None:
      updates |= cu.tracked_update(
          cfg, "stop_after_steps", self.stop_after_steps
      )

    updates |= self._patch_batch_size(cfg)

    if self.skip_eval:
      updates |= cu.tracked_update(cfg, "evals", {})
    else:
      updates |= self._patch_num_batches(cfg)
      updates |= self._patch_eval_run_once(cfg)

    if self.skip_checkpointer:
      updates |= cu.tracked_update(cfg, "checkpointer", None)

    if self.skip_metrics:
      updates |= cu.tracked_update(cfg, "train_metrics", {})

    if self.skip_summaries:
      updates |= cu.tracked_update(cfg, "train_summaries", {})

    if hasattr(cfg, "setup"):
      updates |= cu.tracked_update(
          cfg, "setup.add_flatboard", not self.skip_flatboard
      )

    updates |= self._patch_profile(cfg)
    updates |= self._patch_checkify(cfg)

    return cfg, updates

  def _patch_batch_size(self, cfg: konfig.ConfigDict) -> dict[str, Any]:
    """Patches the batch size."""
    updates = {}
    if self.batch_size is not None:
      batch_size = self.batch_size
      if batch_size == BATCH_SIZE_DEVICES:
        batch_size = jax.local_device_count()

      updates |= cu.tracked_update(cfg, "train_ds.**.batch_size", batch_size)
      updates |= cu.tracked_update(cfg, "train_ds.**.shuffle_buffer_size", 1)
      if hasattr(cfg, "evals"):
        updates |= cu.tracked_update(cfg, "evals.**.batch_size", batch_size)
    return updates

  def _patch_num_batches(self, cfg: konfig.ConfigDict) -> dict[str, Any]:
    """Patches the number of batches."""
    updates = {}
    if self.num_batches:
      if hasattr(cfg, "evals"):
        updates |= cu.tracked_update(
            cfg, "evals.**.num_batches", self.num_batches
        )
    return updates

  def _patch_eval_run_once(self, cfg: konfig.ConfigDict) -> dict[str, Any]:
    """Sets each evaluator to run only once at step 0."""
    updates = {}
    if not self.eval_run_once:
      return updates
    if not hasattr(cfg, "evals"):
      return updates

    run_value = kd_evals.Once(0)  # NOTE: kd_evals is a konfig.import
    for name, eval_cfg in cfg.evals.items():
      eval_cfg.run = run_value
      updates[f"evals.{name}.run"] = run_value
    return updates

  def _patch_profile(self, cfg: konfig.ConfigDict) -> dict[str, Any]:
    """Enables profiling on the config."""
    if not self.profile:
      return {}
    updates = {}
    if not hasattr(cfg, "profiler"):
      # NOTE: kd_inspect is a konfig.import
      cfg.profiler = kd_inspect.Profiler()
      updates["profiler"] = cfg.profiler
    updates |= cu.tracked_update(cfg, "stop_after_steps", 17)
    updates |= cu.tracked_update(cfg, "profiler.profile_duration_ms", None)
    updates |= cu.tracked_update(cfg, "profiler.on_colab", True)
    return updates

  def _patch_checkify(self, cfg: konfig.ConfigDict) -> dict[str, Any]:
    """Enables checkify on the config."""
    if not self.checkify:
      return {}
    # NOTE: jax_checkify is a konfig.import
    return cu.tracked_update(
        cfg, "checkify_error_categories", jax_checkify.all_checks
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigOrigin:
  """Information about the origin of a config (path, overrides, ...)."""

  filename: epath.Path | None = None
  overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
  patches: dict[str, Any] = dataclasses.field(default_factory=dict)

  def summary(self, include_patches: bool = True) -> str:
    """Returns a summary of the config origin as a string."""
    qualifier = "(patched)" if include_patches else "(unpatched)"
    summary = f"========= Config {qualifier} =========\n"
    if self.filename:
      summary += f"Filename: {self.filename}\n"
    if self.overrides:
      summary += f"Overrides:\n {epy.pretty_repr(self.overrides)}\n"
    if include_patches and self.patches:
      summary += f"Patches:\n {epy.pretty_repr(self.patches)}\n"

    return summary
