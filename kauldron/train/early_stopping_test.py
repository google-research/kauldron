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

"""Standalone unit tests for the early-stopping controller.

These tests intentionally stub out JAX / epath / kauldron so the pure state
machine can be exercised without a full Kauldron installation:

    python kauldron/train/early_stopping_test.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _install_stubs() -> None:
  class _NullCtx:
    def __enter__(self):
      return None

    def __exit__(self, *args):
      return False

  jnp = types.ModuleType("jax.numpy")
  jnp.full = lambda shape, value, dtype=None: [value] * shape[0]
  jnp.int32 = "int32"

  jax = types.ModuleType("jax")
  jax.numpy = jnp
  jax.config = types.SimpleNamespace(
      jax_disable_jit=False,
      jax_debug_nans=False,
      update=lambda *a, **k: None,
  )
  jax.lax = types.SimpleNamespace(psum=lambda x, axis: x)
  jax.device_get = lambda x: x
  jax.transfer_guard = lambda mode: _NullCtx()
  jax.local_device_count = lambda: 1
  jax.pmap = lambda fn, axis: fn
  jax.clear_caches = lambda: None
  jax.tree = types.SimpleNamespace(map=lambda fn, tree: None)
  jax.Array = object

  epath = types.ModuleType("etils.epath")
  epath.Path = pathlib.Path
  etils = types.ModuleType("etils")
  etils.epath = epath

  class _Status:
    is_lead_host = True

    def log_status(self, msg):
      pass

  status_mod = types.ModuleType("kauldron.utils.status_utils")
  status_mod.status = _Status()
  kauldron_utils = types.ModuleType("kauldron.utils")
  kauldron_utils.__path__ = []
  kauldron = types.ModuleType("kauldron")
  kauldron.__path__ = []

  sys.modules["jax"] = jax
  sys.modules["jax.numpy"] = jnp
  sys.modules["etils"] = etils
  sys.modules["etils.epath"] = epath
  sys.modules["kauldron"] = kauldron
  sys.modules["kauldron.utils"] = kauldron_utils
  sys.modules["kauldron.utils.status_utils"] = status_mod


_install_stubs()

_spec = importlib.util.spec_from_file_location(
    "kd_early_stopping", os.path.join(_HERE, "early_stopping.py")
)
es = importlib.util.module_from_spec(_spec)
sys.modules["kd_early_stopping"] = es
_spec.loader.exec_module(es)


def _controller(cfg=None, workdir=None):
  return es.EarlyStoppingController.from_config(cfg, workdir)


def _eval(value, name="loss", evaluator="eval"):
  return {evaluator: {name: value}}


class EarlyStoppingTest(unittest.TestCase):

  def test_disabled_is_noop(self):
    c = _controller(None)
    for i in range(5):
      self.assertFalse(c.update(step=i, eval_metrics=_eval(1.0 - i * 0.1)))
    self.assertFalse(c.triggered)
    c.finalize()  # must not raise

  def test_min_mode_patience_and_save_signal(self):
    c = _controller(
        es.EarlyStopping(
            monitor="eval.loss", mode="min", patience=2,
            sync_across_hosts=False,
        )
    )
    values = [1.0, 0.9, 0.95, 0.96, 0.97]
    saves = [
        c.update(step=i, eval_metrics=_eval(v))
        for i, v in enumerate(values, start=1)
    ]
    self.assertEqual(c.state.best_value, 0.9)
    self.assertEqual(c.state.best_step, 2)
    self.assertTrue(c.triggered)
    self.assertEqual(c.state.stopped_at_step, 5)
    self.assertEqual(c.state.bad_points, 3)
    self.assertEqual(saves, [False, False, False, False, True])

  def test_max_mode_auto_and_zero_patience(self):
    c = _controller(
        es.EarlyStopping(
            monitor="eval.accuracy", mode="auto", patience=0,
            sync_across_hosts=False,
        )
    )
    self.assertEqual(c.cfg.resolved_mode, "max")
    c.update(step=1, eval_metrics=_eval(0.5, name="accuracy"))
    triggered = c.update(step=2, eval_metrics=_eval(0.4, name="accuracy"))
    self.assertTrue(triggered)
    self.assertTrue(c.triggered)

  def test_min_delta(self):
    c = _controller(
        es.EarlyStopping(
            monitor="eval.loss", mode="min", min_delta=0.01, patience=0,
            sync_across_hosts=False,
        )
    )
    c.update(step=1, eval_metrics=_eval(0.9))
    # 0.001 improvement is below min_delta -> counts as a bad point.
    self.assertTrue(c.update(step=2, eval_metrics=_eval(0.899)))

  def test_warmup_tracks_best_without_stopping(self):
    c = _controller(
        es.EarlyStopping(
            monitor="eval.loss", mode="min", min_steps=5, patience=0,
            sync_across_hosts=False,
        )
    )
    for step in range(1, 5):
      self.assertFalse(c.update(step=step, eval_metrics=_eval(float(step))))
    self.assertEqual(c.state.best_value, 1.0)
    self.assertEqual(c.state.bad_points, 0)
    self.assertTrue(c.update(step=5, eval_metrics=_eval(5.0)))

  def test_steps_without_eval_do_not_consume_patience(self):
    c = _controller(
        es.EarlyStopping(
            monitor="eval.loss", mode="min", patience=1,
            sync_across_hosts=False,
        )
    )
    c.update(step=1, eval_metrics=_eval(1.0))
    for step in range(2, 10):
      c.update(step=step, eval_metrics={})
    self.assertEqual(c.state.num_points, 1)
    self.assertEqual(c.state.bad_points, 0)
    self.assertFalse(c.triggered)

  def test_non_finite_triggers_immediately(self):
    c = _controller(
        es.EarlyStopping(monitor="eval.loss", sync_across_hosts=False)
    )
    self.assertTrue(c.update(step=1, eval_metrics=_eval(float("nan"))))
    self.assertIn("non-finite", c.reason)

  def test_slash_path_and_unique_leaf_fallback(self):
    c = _controller(
        es.EarlyStopping(
            monitor="eval/loss", mode="min", patience=0,
            sync_across_hosts=False,
        )
    )
    # Nested `metric_values`: the leaf-name fallback resolves `eval.loss`.
    c.update(
        step=1,
        eval_metrics={"eval": {"losses": {"loss": 0.5}}},
    )
    self.assertEqual(c.state.best_value, 0.5)
    self.assertTrue(
        c.update(step=2, eval_metrics={"eval": {"losses": {"loss": 0.6}}})
    )

  def test_missing_monitor_lists_available_keys(self):
    c = _controller(
        es.EarlyStopping(monitor="eval.nope", sync_across_hosts=False)
    )
    with self.assertRaises(KeyError) as ctx:
      c.update(step=1, eval_metrics=_eval(1.0))
    self.assertIn("Available metric keys", str(ctx.exception))
    self.assertIn("eval.loss", str(ctx.exception))

  def test_train_monitor_uses_train_metrics(self):
    cfg = es.EarlyStopping(
        monitor="train.loss", mode="min", patience=0,
        sync_across_hosts=False,
    )
    self.assertTrue(cfg.needs_train_metrics)
    c = _controller(cfg)
    self.assertTrue(c.needs_train_metrics)
    c.update(step=1, eval_metrics={}, train_metrics={"loss": 1.0})
    self.assertTrue(
        c.update(step=2, eval_metrics={}, train_metrics={"loss": 1.1})
    )

  def test_baseline_does_not_consume_patience(self):
    c = _controller(
        es.EarlyStopping(
            monitor="eval.accuracy", mode="max", baseline=0.9, patience=0,
            sync_across_hosts=False,
        )
    )
    for step, value in enumerate([0.5, 0.6, 0.7], start=1):
      self.assertFalse(
          c.update(step=step, eval_metrics=_eval(value, name="accuracy"))
      )
    self.assertEqual(c.state.bad_points, 0)
    # Reaching the baseline resets the counter, the next bad point stops.
    c.update(step=4, eval_metrics=_eval(0.95, name="accuracy"))
    self.assertTrue(
        c.update(step=5, eval_metrics=_eval(0.94, name="accuracy"))
    )

  def test_state_persists_and_resumes(self):
    with tempfile.TemporaryDirectory() as tmp:
      cfg = es.EarlyStopping(
          monitor="eval.loss", mode="min", patience=1,
          sync_across_hosts=False,
      )
      c = _controller(cfg, tmp)
      for step, value in enumerate([1.0, 0.8, 0.85, 0.86], start=1):
        c.update(step=step, eval_metrics=_eval(value))
      self.assertTrue(c.triggered)
      c.finalize()

      state_path = pathlib.Path(tmp) / es.STATE_FILENAME
      self.assertTrue(state_path.exists())
      saved = json.loads(state_path.read_text())
      self.assertEqual(saved["best_value"], 0.8)
      self.assertTrue((pathlib.Path(tmp) / es.EARLY_STOP_FILENAME).exists())

      resumed = _controller(cfg, tmp)
      resumed.maybe_restore()
      self.assertEqual(resumed.state.best_value, 0.8)
      self.assertEqual(resumed.state.best_step, 2)
      self.assertFalse(resumed.triggered)  # terminal flag is cleared
      self.assertEqual(resumed.state.bad_points, 2)

  def test_invalid_config(self):
    with self.assertRaises(ValueError):
      es.EarlyStopping(mode="bigger")
    with self.assertRaises(ValueError):
      es.EarlyStopping(patience=-1)
    with self.assertRaises(ValueError):
      es.EarlyStopping(monitor="")


if __name__ == "__main__":
  unittest.main(verbosity=2)
