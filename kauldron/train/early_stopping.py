# Copyright 2026 The kauldron Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional early stopping for the Kauldron training loop.

Design notes
------------
* The controller only consumes metrics that ``train_impl`` already computes
  (evaluation metrics from the *old* model state, plus optional train
  metrics). It never reorders the operations in the training loop.
* Only steps on which at least one evaluator produced metrics count as an
  "evaluation point" for ``patience``. Steps without an eval are ignored.
* On evaluated steps the trigger flag is all-reduced across hosts, so every
  host leaves the loop on the same step. A mismatch here would dead-lock in
  ``train_loop._sync()``.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import math
from typing import Any, Optional

from etils import epath
import jax
import jax.numpy as jnp
import numpy as np
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

EARLY_STOP_FILENAME = "early_stopping_triggered"
STATE_FILENAME = "early_stopping_state.json"
_MAX_HISTORY = 50
_MAX_LOGGED_KEYS = 30

_MAX_KEYS_FOR_BETTER = (
    "acc",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auc",
    "score",
    "bleu",
    "psnr",
    "ssim",
    "top1",
    "top5",
    "r2",
)


@dataclasses.dataclass(frozen=True)
class EarlyStopping:
  """Configuration object; assign it to ``Trainer.early_stopping``.

  Attributes:
    monitor: Dotted (or slash-separated) path into the metrics mapping built by
      ``train_impl``. Eval results live under the evaluator name (default
      ``"eval"``), e.g. ``"eval.accuracy"``. Train metrics, when requested,
      live under ``"train"``. The first evaluated step logs all available keys.
    mode: ``"min"``, ``"max"`` or ``"auto"`` (guess from the metric name).
    min_delta: Minimum absolute change that counts as an improvement.
    patience: Allowed consecutive non-improving evaluation points. ``0`` stops
      at the first non-improvement.
    min_steps: Warm-up; never trigger before this global step.
    baseline: Optional value that must be reached before patience is consumed.
    check_finite: Stop immediately on NaN / inf.
    save_on_trigger: Force one checkpoint save on the triggering step. The
      saved state is the same *old* state the triggering eval metrics describe.
    sync_across_hosts: All-reduce the trigger flag on evaluated steps. Keep it
      on for multi-host runs; single-host runs may disable it.
    log_available_metrics: Log the flattened metric keys on the first eval.
  """

  monitor: str = "eval.loss"
  mode: str = "auto"
  min_delta: float = 0.0
  patience: int = 3
  min_steps: int = 0
  baseline: Optional[float] = None
  check_finite: bool = True
  save_on_trigger: bool = True
  sync_across_hosts: bool = True
  log_available_metrics: bool = True

  def __post_init__(self) -> None:
    if self.mode not in ("min", "max", "auto"):
      raise ValueError(
          f"`mode` must be 'min', 'max' or 'auto', got {self.mode!r}"
      )
    if self.min_delta < 0:
      raise ValueError("`min_delta` must be >= 0.")
    if self.patience < 0:
      raise ValueError("`patience` must be >= 0.")
    if self.min_steps < 0:
      raise ValueError("`min_steps` must be >= 0.")
    if not self.monitor:
      raise ValueError("`monitor` must not be empty.")

  @property
  def needs_train_metrics(self) -> bool:
    return _normalize(self.monitor).split(".")[0] == "train"

  @property
  def resolved_mode(self) -> str:
    if self.mode != "auto":
      return self.mode
    name = _normalize(self.monitor)
    return "max" if any(k in name for k in _MAX_KEYS_FOR_BETTER) else "min"


@dataclasses.dataclass
class _State:
  best_value: Optional[float] = None
  best_step: Optional[int] = None
  bad_points: int = 0
  num_points: int = 0
  triggered: bool = False
  reason: str = ""
  stopped_at_step: Optional[int] = None
  last_step: Optional[int] = None
  history: list[dict[str, Any]] = dataclasses.field(default_factory=list)


class EarlyStoppingController:
  """Stateful wrapper around :class:`EarlyStopping`.

  A disabled controller (``cfg is None``) is a cheap no-op: ``update()``
  returns ``False`` and ``finalize()`` does nothing. This keeps the default
  training path byte-for-byte equivalent to the original loop.
  """

  def __init__(self, cfg: Optional[EarlyStopping], workdir: Any) -> None:
    self.cfg = cfg
    self.enabled = cfg is not None
    if workdir is None:
      self.workdir = None
    elif isinstance(workdir, epath.Path):
      self.workdir = workdir
    else:
      self.workdir = epath.Path(workdir)
    self.state = _State()
    self._logged_keys = False

  @classmethod
  def from_config(
      cls, cfg: Optional[EarlyStopping], workdir: Any
  ) -> EarlyStoppingController:
    return cls(cfg, workdir)

  @property
  def triggered(self) -> bool:
    return self.state.triggered

  @property
  def reason(self) -> str:
    return self.state.reason

  @property
  def needs_train_metrics(self) -> bool:
    return self.enabled and self.cfg.needs_train_metrics

  def maybe_restore(self) -> None:
    """Restore patience / best value when resuming from a checkpoint."""
    if not self.enabled or self.workdir is None or not self.workdir.exists():
      return
    path = self.workdir.joinpath(STATE_FILENAME)
    if not path.exists():
      return
    try:
      with path.open("r") as f:
        data = json.load(f)
    except (OSError, ValueError) as e:
      status.log_status(
          f"early_stopping: could not restore state ({e}); starting fresh."
      )
      return

    defaults = {
        f.name: getattr(_State(), f.name) for f in dataclasses.fields(_State)
    }
    defaults.update({k: v for k, v in data.items() if k in defaults})
    # A run that stopped on purpose must be able to resume: keep the best
    # value and the patience counters, but clear the terminal flag.
    defaults["triggered"] = False
    defaults["reason"] = ""
    defaults["stopped_at_step"] = None
    self.state = _State(**defaults)
    status.log_status(
        f"early_stopping: restored best={self.state.best_value} at step "
        f"{self.state.best_step}, bad_points={self.state.bad_points}."
    )

  def update(
      self,
      *,
      step: int,
      eval_metrics: Mapping[str, Any],
      train_metrics: Any = None,
  ) -> bool:
    """Consume one training step.

    Args:
      step: Current global step.
      eval_metrics: ``{evaluator.name: metric_values}`` produced by the evals
        that ran on this step. Empty when no evaluator ran.
      train_metrics: Optional ``metric_values`` of the previous train step.
        Only supplied when ``monitor`` points into ``train``.

    Returns:
      Whether ``train_impl`` must force a checkpoint save on this step.
    """
    if not self.enabled:
      return False

    was_triggered = self.state.triggered
    self.state.last_step = int(step)

    metrics: dict[str, Any] = dict(eval_metrics)
    if train_metrics is not None:
      metrics["train"] = train_metrics

    has_eval_point = bool(metrics)
    if (
        has_eval_point
        and self.cfg.log_available_metrics
        and not self._logged_keys
    ):
      self._logged_keys = True
      status.log_status(
          "early_stopping: available metric keys: "
          + ", ".join(sorted(_flatten(metrics))[:_MAX_LOGGED_KEYS])
      )

    reason = ""
    triggered_here = False
    if has_eval_point:
      scalar = _to_float(_resolve_metric(metrics, self.cfg.monitor))
      self.state.num_points += 1

      if self.cfg.check_finite and not math.isfinite(scalar):
        triggered_here = True
        reason = f"non-finite {self.cfg.monitor}={scalar}"
      elif int(step) < self.cfg.min_steps:
        # Warm-up: still track the best value, but never stop nor consume
        # patience, so the post-warmup baseline is meaningful.
        if self._is_improvement(scalar):
          self.state.best_value = scalar
          self.state.best_step = int(step)
        self.state.bad_points = 0
      else:
        if self._is_improvement(scalar):
          self.state.best_value = scalar
          self.state.best_step = int(step)
          self.state.bad_points = 0
        elif self.cfg.baseline is not None and not self._reached_baseline(
            scalar
        ):
          # Still waiting to reach the required baseline: do not spend
          # patience while the model is simply not good enough yet.
          self.state.bad_points = 0
        else:
          self.state.bad_points += 1
          if self.state.bad_points > self.cfg.patience:
            triggered_here = True
            reason = (
                f"{self.cfg.monitor} did not improve for "
                f"{self.state.bad_points} evaluation points "
                f"(patience={self.cfg.patience}, best="
                f"{self.state.best_value} at step {self.state.best_step})"
            )

      self._append_history(
          {
              "step": int(step),
              "value": scalar,
              "best": self.state.best_value,
              "bad_points": self.state.bad_points,
              "triggered": triggered_here,
          }
      )
      if triggered_here and not self.state.triggered:
        self.state.triggered = True
        self.state.reason = reason
        self.state.stopped_at_step = int(step)

    if self.cfg.sync_across_hosts and has_eval_point:
      # All hosts execute this collective on exactly the same (evaluated)
      # steps, so a trigger raised anywhere propagates everywhere.
      if _any_host_triggered(self.state.triggered) and not self.state.triggered:
        self.state.triggered = True
        self.state.reason = "triggered on another host"
        self.state.stopped_at_step = int(step)

    newly_triggered = self.state.triggered and not was_triggered
    return bool(newly_triggered and self.cfg.save_on_trigger)

  def finalize(self) -> None:
    """Persist the state and mark an early-stopped run (lead host only)."""
    if not self.enabled:
      return
    self._write_state()
    if (
        self.state.triggered
        and self.workdir is not None
        and self.workdir.exists()
        and status.is_lead_host
    ):
      self.workdir.joinpath(EARLY_STOP_FILENAME).touch()

  def _is_improvement(self, value: float) -> bool:
    if self.state.best_value is None:
      return True
    if self.cfg.resolved_mode == "max":
      return value - self.state.best_value > self.cfg.min_delta
    return self.state.best_value - value > self.cfg.min_delta

  def _reached_baseline(self, value: float) -> bool:
    if self.cfg.resolved_mode == "max":
      return value >= self.cfg.baseline
    return value <= self.cfg.baseline

  def _append_history(self, entry: dict[str, Any]) -> None:
    self.state.history.append(entry)
    if len(self.state.history) > _MAX_HISTORY:
      self.state.history = self.state.history[-_MAX_HISTORY:]

  def _write_state(self) -> None:
    if (
        not self.enabled
        or self.workdir is None
        or not self.workdir.exists()
        or not status.is_lead_host
    ):
      return
    path = self.workdir.joinpath(STATE_FILENAME)
    with path.open("w") as f:
      json.dump(dataclasses.asdict(self.state), f, indent=2, sort_keys=True)


def _normalize(key: str) -> str:
  return key.replace("/", ".").strip(".").lower()


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
  """Flatten nested mappings / dataclasses into ``dotted.key`` -> leaf."""
  out: dict[str, Any] = {}
  if isinstance(obj, Mapping):
    for k, v in obj.items():
      key = f"{prefix}.{k}" if prefix else str(k)
      out.update(_flatten(v, key))
  elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    for f in dataclasses.fields(obj):
      key = f"{prefix}.{f.name}" if prefix else f.name
      out.update(_flatten(getattr(obj, f.name), key))
  elif prefix:
    out[prefix] = obj
  return out


def _resolve_metric(metrics: Mapping[str, Any], key: str) -> Any:
  """Resolve ``key`` in a (possibly nested) metric mapping.

  Accepts ``/`` and ``.`` separators, case-insensitive full paths, and falls
  back to a unique leaf-name match (handy when ``metric_values`` nests under
  ``losses`` / ``metrics``).
  """
  flat = _flatten(metrics)
  if not flat:
    raise KeyError("early_stopping: no metric values available at this step.")
  target = _normalize(key)

  for k, v in flat.items():
    if _normalize(k) == target:
      return v

  leaf = target.split(".")[-1]
  candidates = [
      (k, v) for k, v in flat.items() if _normalize(k).split(".")[-1] == leaf
  ]
  if len(candidates) == 1:
    return candidates[0][1]

  keys = ", ".join(sorted(flat)[:_MAX_LOGGED_KEYS])
  if len(candidates) > 1:
    raise KeyError(
        f"early_stopping: monitor={key!r} is ambiguous; matches "
        f"{[k for k, _ in candidates]}. Use a full path. Available: {keys}"
    )
  raise KeyError(
      f"early_stopping: monitor={key!r} not found. Available metric keys: "
      f"{keys}. Set `monitor` to one of these."
  )


def _to_float(value: Any) -> float:
  with jax.transfer_guard("allow"):
    host_value = jax.device_get(value)
  arr = np.asarray(host_value)
  if arr.size != 1:
    raise ValueError(
        f"early_stopping: monitored value must be a scalar, got shape "
        f"{arr.shape}."
    )
  return float(arr.reshape(-1)[0])


def _any_host_triggered(triggered: bool) -> bool:
  """All-reduce a boolean across every host/device."""
  x = jnp.full(
      [jax.local_device_count()], 1 if triggered else 0, dtype=jnp.int32
  )

  def _psync(t):
    return jax.lax.psum(t, "i")

  with jax.transfer_guard("allow"):
    y = jax.pmap(_psync, "i")(x)
  return int(jax.device_get(y)[0]) > 0
