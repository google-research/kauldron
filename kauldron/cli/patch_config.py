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

import dataclasses
from typing import Any

from etils import epath
from etils import epy
from kauldron import konfig
from kauldron import kontext


@dataclasses.dataclass(frozen=True, kw_only=True)
class PatchConfig:
  """Patches a config, in order to speed up test runs."""

  stop_after_steps: int | None = 1
  """Stop training after N steps. Defaults to 1."""

  batch_size: int | str | None = "devices"
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

  def __call__(self, cfg: konfig.ConfigDict) -> konfig.ConfigDict:
    """Returns a mutated config."""
    if self.stop_after_steps is not None:
      cfg.stop_after_steps = self.stop_after_steps

    self._patch_batch_size(cfg)

    if self.skip_eval:
      cfg.evals = {}
    else:
      self._patch_num_batches(cfg)
      self._patch_eval_run_once(cfg)

    if self.skip_checkpointer:
      cfg.checkpointer = None

    if self.skip_metrics:
      cfg.train_metrics = {}

    if self.skip_summaries:
      cfg.train_summaries = {}

    if hasattr(cfg, "setup"):
      cfg.setup.add_flatboard = not self.skip_flatboard

    self._patch_profile(cfg)
    self._patch_checkify(cfg)

    return cfg

  def _patch_batch_size(self, cfg: konfig.ConfigDict) -> None:
    """Patches the batch size."""
    if self.batch_size is not None:
      batch_size = self.batch_size
      if batch_size == "devices":
        import jax  # pylint: disable=g-import-not-at-top

        batch_size = jax.local_device_count()

      kontext.set_by_path(cfg, "train_ds.**.batch_size", batch_size)
      kontext.set_by_path(cfg, "train_ds.**.shuffle_buffer_size", 1)
      if hasattr(cfg, "evals"):
        kontext.set_by_path(cfg, "evals.**.batch_size", batch_size)

  def _patch_num_batches(self, cfg: konfig.ConfigDict) -> None:
    """Patches the number of batches."""
    if self.num_batches is not None:
      if hasattr(cfg, "evals"):
        # TODO(klausg): also set for Evaluators without explicit num_batches.
        # Basically if num_batches is left at None (the default),
        # then it won't be part of the config, but we should still set it here.
        kontext.set_by_path(cfg, "evals.**.num_batches", self.num_batches)

  def _patch_eval_run_once(self, cfg: konfig.ConfigDict) -> None:
    """Sets each evaluator to run only once at step 0."""
    if not self.eval_run_once:
      return
    if not hasattr(cfg, "evals"):
      return
    with konfig.imports(lazy=True):
      from kauldron import evals as kd_evals  # pylint: disable=g-import-not-at-top

    for _, eval_cfg in cfg.evals.items():
      eval_cfg.run = kd_evals.Once(0)

  def _patch_profile(self, cfg: konfig.ConfigDict) -> None:
    """Enables profiling on the config."""
    if not self.profile:
      return
    if not hasattr(cfg, "profiler"):
      with konfig.imports(lazy=True):
        from kauldron import inspect as kd_inspect  # pylint: disable=g-import-not-at-top

      cfg.profiler = kd_inspect.Profiler()
    cfg.stop_after_steps = 17  # Profiling is at step 10
    cfg.profiler.profile_duration_ms = None
    cfg.profiler.on_colab = True

  def _patch_checkify(self, cfg: konfig.ConfigDict) -> None:
    """Enables checkify on the config."""
    if not self.checkify:
      return
    with konfig.imports(lazy=True):
      from jax.experimental import checkify as jax_checkify  # pylint: disable=g-import-not-at-top

      cfg.checkify_error_categories = jax_checkify.all_checks


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigOrigin:
  """Information about the origin of a config (path, overrides, ...)."""

  filename: epath.Path | None = None
  overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
  patches: dict[str, Any] = dataclasses.field(default_factory=dict)

  def summary(self) -> str:
    """Returns a summary of the config origin as a string."""
    summary = "========= Config =========\n"
    if self.filename:
      summary += f"Filename: {self.filename}\n"
    if self.overrides:
      summary += f"Overrides:\n {epy.pretty_repr(self.overrides)}\n"
    if self.patches:
      summary += f"Patches:\n {epy.pretty_repr(self.patches)}\n"
    summary += "==========================\n"
    return summary
