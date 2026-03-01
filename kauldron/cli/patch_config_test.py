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
    result = patch_config.PatchConfig()(cfg)
    assert result.stop_after_steps == 1
    assert cfg.train_ds.batch_size == 2
    assert cfg.train_ds.shuffle_buffer_size == 1
    assert cfg.evals.eval.batch_size == 2
    assert cfg.evals.eval.num_batches == 1
    assert cfg.checkpointer is None
    assert cfg.train_metrics.acc == "accuracy"
    assert cfg.train_summaries.img == "image"


class TestPatchConfigBatchSize:

  def test_explicit_integer(self):
    cfg = _make_cfg(checkpointer=None)
    patch_config.PatchConfig(batch_size=32)(cfg)
    assert cfg.train_ds.batch_size == 32
    assert cfg.train_ds.shuffle_buffer_size == 1

  def test_none_leaves_unchanged(self):
    cfg = _make_cfg(checkpointer=None)
    patch_config.PatchConfig(batch_size=None)(cfg)
    assert cfg.train_ds.batch_size == 64
    assert cfg.train_ds.shuffle_buffer_size == 1000

  def test_propagates_to_evals(self):
    cfg = _make_cfg(
        evals={"eval": {"batch_size": 128, "num_batches": 10}},
        checkpointer=None,
    )
    patch_config.PatchConfig(batch_size=8)(cfg)
    assert cfg.evals.eval.batch_size == 8


class TestPatchConfigSkipFlags:

  @_DEVICE_COUNT
  def test_skip_eval_clears_evals(self, _):
    cfg = _make_cfg(evals={"eval": {"batch_size": 128}}, checkpointer=None)
    patch_config.PatchConfig(skip_eval=True)(cfg)
    assert len(cfg.evals) == 0

  @_DEVICE_COUNT
  def test_skip_metrics(self, _):
    cfg = _make_cfg(
        checkpointer=None,
        train_metrics={"acc": "accuracy"},
    )
    patch_config.PatchConfig(skip_metrics=True)(cfg)
    assert len(cfg.train_metrics) == 0

  @_DEVICE_COUNT
  def test_skip_summaries(self, _):
    cfg = _make_cfg(
        checkpointer=None,
        train_summaries={"img": "image"},
    )
    patch_config.PatchConfig(skip_summaries=True)(cfg)
    assert len(cfg.train_summaries) == 0

  @_DEVICE_COUNT
  def test_skip_flatboard(self, _):
    cfg = _make_cfg(checkpointer=None, setup={"add_flatboard": True})
    patch_config.PatchConfig(skip_flatboard=True)(cfg)
    assert not cfg.setup.add_flatboard

  @_DEVICE_COUNT
  def test_no_skip_flatboard(self, _):
    cfg = _make_cfg(checkpointer=None, setup={"add_flatboard": False})
    patch_config.PatchConfig(skip_flatboard=False)(cfg)
    assert cfg.setup.add_flatboard

  def test_stop_after_steps_none_leaves_unchanged(self):
    cfg = _make_cfg(checkpointer=None)
    with _DEVICE_COUNT:
      patch_config.PatchConfig(stop_after_steps=None)(cfg)
    assert cfg.stop_after_steps == 100

  def test_num_batches_propagates_to_evals(self):
    cfg = _make_cfg(
        evals={"eval": {"batch_size": 128, "num_batches": 10}},
        checkpointer=None,
    )
    patch_config.PatchConfig(batch_size=8, num_batches=3)(cfg)
    assert cfg.evals.eval.num_batches == 3


class TestPatchConfigProfile:

  @_DEVICE_COUNT
  def test_profile_sets_profiler_and_steps(self, _):
    cfg = _make_cfg(checkpointer=None)
    patch_config.PatchConfig(profile=True)(cfg)
    assert cfg.stop_after_steps == 17
    assert hasattr(cfg, "profiler")
    assert cfg.profiler.profile_duration_ms is None
    assert cfg.profiler.on_colab


class TestPatchConfigCheckify:

  @_DEVICE_COUNT
  def test_checkify_sets_error_categories(self, _):
    cfg = _make_cfg(checkpointer=None)
    patch_config.PatchConfig(checkify=True)(cfg)
    assert hasattr(cfg, "checkify_error_categories")


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
