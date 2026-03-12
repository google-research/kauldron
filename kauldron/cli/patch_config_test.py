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

from __future__ import annotations

from unittest import mock

from etils import epath
from kauldron import konfig
from kauldron.cli import patch_config


_DEVICE_COUNT = mock.patch("jax.local_device_count", return_value=2)


def _make_cfg(**overrides) -> konfig.ConfigDict:
  base = {
      "stop_after_steps": 100,
      "train_ds": {"batch_size": 64, "shuffle_buffer_size": 1000},
      "checkpointer": {"save_interval_steps": 10},
  }
  base.update(overrides)
  return konfig.ConfigDict(base)


class TestPatchConfigDefaults:

  @_DEVICE_COUNT
  def test_all_defaults(self, _):
    cfg = _make_cfg(
        evals={"eval": {"batch_size": 128, "num_batches": 10}},
        train_metrics={"acc": "accuracy"},
        train_summaries={"img": "image"},
    )
    patched, _ = patch_config.PatchConfig()(cfg)
    assert patched.stop_after_steps == 1
    assert patched.train_ds.batch_size == 2
    assert patched.train_ds.shuffle_buffer_size == 1
    assert patched.evals.eval.batch_size == 2
    assert patched.evals.eval.num_batches == 1
    assert patched.checkpointer is None
    assert patched.train_metrics.acc == "accuracy"
    assert patched.train_summaries.img == "image"


class TestPatchConfigBatchSize:

  def test_explicit_integer(self):
    cfg = _make_cfg(checkpointer=None)
    patched, _ = patch_config.PatchConfig(batch_size=32)(cfg)
    assert patched.train_ds.batch_size == 32
    assert patched.train_ds.shuffle_buffer_size == 1

  def test_none_leaves_unchanged(self):
    cfg = _make_cfg(checkpointer=None)
    patched, _ = patch_config.PatchConfig(batch_size=None)(cfg)
    assert patched.train_ds.batch_size == 64
    assert patched.train_ds.shuffle_buffer_size == 1000

  def test_propagates_to_evals(self):
    cfg = _make_cfg(
        evals={"eval": {"batch_size": 128, "num_batches": 10}},
        checkpointer=None,
    )
    patched, _ = patch_config.PatchConfig(batch_size=8)(cfg)
    assert patched.evals.eval.batch_size == 8


class TestPatchConfigSkipFlags:

  @_DEVICE_COUNT
  def test_skip_eval_clears_evals(self, _):
    cfg = _make_cfg(evals={"eval": {"batch_size": 128}}, checkpointer=None)
    patched, _ = patch_config.PatchConfig(skip_eval=True)(cfg)
    assert len(patched.evals) == 0

  @_DEVICE_COUNT
  def test_skip_metrics(self, _):
    cfg = _make_cfg(
        checkpointer=None,
        train_metrics={"acc": "accuracy"},
    )
    patched, _ = patch_config.PatchConfig(skip_metrics=True)(cfg)
    assert len(patched.train_metrics) == 0

  @_DEVICE_COUNT
  def test_skip_summaries(self, _):
    cfg = _make_cfg(
        checkpointer=None,
        train_summaries={"img": "image"},
    )
    patched, _ = patch_config.PatchConfig(skip_summaries=True)(cfg)
    assert len(patched.train_summaries) == 0

  @_DEVICE_COUNT
  def test_skip_flatboard(self, _):
    cfg = _make_cfg(checkpointer=None, setup={"add_flatboard": True})
    patched, _ = patch_config.PatchConfig(skip_flatboard=True)(cfg)
    assert not patched.setup.add_flatboard

  @_DEVICE_COUNT
  def test_no_skip_flatboard(self, _):
    cfg = _make_cfg(checkpointer=None, setup={"add_flatboard": False})
    patched, _ = patch_config.PatchConfig(skip_flatboard=False)(cfg)
    assert patched.setup.add_flatboard

  def test_stop_after_steps_none_leaves_unchanged(self):
    cfg = _make_cfg(checkpointer=None)
    with _DEVICE_COUNT:
      patched, _ = patch_config.PatchConfig(stop_after_steps=None)(cfg)
    assert patched.stop_after_steps == 100

  def test_num_batches_propagates_to_evals(self):
    cfg = _make_cfg(
        evals={"eval": {"batch_size": 128, "num_batches": 10}},
        checkpointer=None,
    )
    patched, _ = patch_config.PatchConfig(batch_size=8, num_batches=3)(cfg)
    assert patched.evals.eval.num_batches == 3


class TestPatchConfigProfile:

  @_DEVICE_COUNT
  def test_profile_sets_profiler_and_steps(self, _):
    cfg = _make_cfg(checkpointer=None)
    patched, _ = patch_config.PatchConfig(profile=True)(cfg)
    assert patched.stop_after_steps == 17
    assert hasattr(patched, "profiler")
    assert patched.profiler.profile_duration_ms is None
    assert patched.profiler.on_colab


class TestPatchConfigCheckify:

  @_DEVICE_COUNT
  def test_checkify_sets_error_categories(self, _):
    cfg = _make_cfg(checkpointer=None)
    patched, _ = patch_config.PatchConfig(checkify=True)(cfg)
    assert hasattr(patched, "checkify_error_categories")


class TestPatchConfigUpdates:

  @_DEVICE_COUNT
  def test_default_updates_include_batch_and_eval_paths(self, _):
    cfg = _make_cfg(
        evals={"eval": {"batch_size": 128, "num_batches": 10}},
    )
    _, updates = patch_config.PatchConfig()(cfg)
    assert "stop_after_steps" in updates
    assert "checkpointer" in updates
    assert "train_ds.batch_size" in updates
    assert "train_ds.shuffle_buffer_size" in updates
    assert "evals.eval.batch_size" in updates
    assert "evals.eval.num_batches" in updates
    assert "evals.eval.run" in updates

  @_DEVICE_COUNT
  def test_skip_eval_excludes_eval_sub_paths(self, _):
    cfg = _make_cfg(
        evals={"eval": {"batch_size": 128, "num_batches": 10}},
    )
    _, updates = patch_config.PatchConfig(skip_eval=True)(cfg)
    assert not updates["evals"]
    assert "evals.eval.num_batches" not in updates
    assert "evals.eval.run" not in updates

  @_DEVICE_COUNT
  def test_batch_size_none_excludes_batch_paths(self, _):
    cfg = _make_cfg()
    _, updates = patch_config.PatchConfig(batch_size=None)(cfg)
    assert "train_ds.batch_size" not in updates
    assert "train_ds.shuffle_buffer_size" not in updates

  @_DEVICE_COUNT
  def test_profile_updates(self, _):
    cfg = _make_cfg(checkpointer=None)
    _, updates = patch_config.PatchConfig(profile=True)(cfg)
    assert "profiler" in updates
    assert updates["stop_after_steps"] == 17
    assert updates["profiler.profile_duration_ms"] is None
    assert updates["profiler.on_colab"]

  @_DEVICE_COUNT
  def test_checkify_updates(self, _):
    cfg = _make_cfg(checkpointer=None)
    _, updates = patch_config.PatchConfig(checkify=True)(cfg)
    assert "checkify_error_categories" in updates


class TestConfigOrigin:

  def test_summary_with_all_fields(self):
    origin = patch_config.ConfigOrigin(
        filename=epath.Path("/path/to/config.py"),
        overrides={"seed": 42},
        patches={"workdir": "/tmp/test"},
    )
    s = origin.summary()
    assert "Filename:" in s
    assert "config.py" in s
    assert "Overrides" in s
    assert "seed" in s
    assert "Patches" in s
    assert "workdir" in s

  def test_summary_with_defaults(self):
    origin = patch_config.ConfigOrigin()
    s = origin.summary()
    assert "Config" in s
    assert "Filename" not in s
    assert "Overrides" not in s
    assert "Patches" not in s

  def test_summary_with_filename_only(self):
    origin = patch_config.ConfigOrigin(
        filename=epath.Path("/path/to/config.py")
    )
    s = origin.summary()
    assert "Filename:" in s
    assert "config.py" in s
    assert "Overrides" not in s
    assert "Patches" not in s
