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

"""Tests for metric_writer refactored write_step_metrics."""

from unittest import mock

from kauldron import summaries
from kauldron.train import metric_writer
import numpy as np
import pytest


class _RecordingWriter:
  """Fake writer that records calls to write_* methods for testing."""

  def __init__(self):
    self.calls = {}

  def write_scalars(self, step, scalars):
    self.calls.setdefault("scalars", []).append((step, dict(scalars)))

  def write_images(self, step, images):
    self.calls.setdefault("images", []).append((step, dict(images)))

  def write_histograms(self, step, arrays, num_buckets=None):
    self.calls.setdefault("histograms", []).append(
        (step, dict(arrays), dict(num_buckets or {}))
    )

  def write_texts(self, step, texts):
    self.calls.setdefault("texts", []).append((step, dict(texts)))

  def write_pointcloud(
      self, step, point_clouds, *, point_colors=None, configs=None
  ):
    del point_colors, configs  # Unused
    self.calls.setdefault("pointcloud", []).append((step, dict(point_clouds)))


@pytest.fixture(name="writer")
def writer_fixture():
  return _RecordingWriter()


class TestWriteValuesByType:

  def test_dispatch_scalars(self, writer):
    values = {"loss": 0.5, "accuracy": 0.9}
    metric_writer._write_values_by_type(writer, step=10, values=values)
    assert "scalars" in writer.calls
    assert writer.calls["scalars"][0] == (10, {"loss": 0.5, "accuracy": 0.9})

  def test_dispatch_images(self, writer):
    img = np.zeros((2, 8, 8, 3), dtype=np.float32)
    values = {"my_image": img}
    metric_writer._write_values_by_type(writer, step=5, values=values)
    assert "images" in writer.calls
    assert writer.calls["images"][0][0] == 5
    np.testing.assert_array_equal(writer.calls["images"][0][1]["my_image"], img)

  def test_dispatch_histograms(self, writer):
    hist = summaries.Histogram(tensor=np.array([1.0, 2.0, 3.0]), num_buckets=10)
    values = {"my_hist": hist}
    metric_writer._write_values_by_type(writer, step=7, values=values)
    assert "histograms" in writer.calls
    assert writer.calls["histograms"][0][2] == {"my_hist": 10}

  def test_dispatch_texts(self, writer):
    values = {"msg": "hello world"}
    metric_writer._write_values_by_type(writer, step=1, values=values)
    assert "texts" in writer.calls
    assert writer.calls["texts"][0] == (1, {"msg": "hello world"})

  def test_dispatch_mixed(self, writer):
    values = {
        "loss": 1.0,
        "msg": "hello",
        "img": np.zeros((1, 4, 4, 3), dtype=np.float32),
    }
    metric_writer._write_values_by_type(writer, step=3, values=values)
    assert "scalars" in writer.calls
    assert "texts" in writer.calls
    assert "images" in writer.calls
    assert writer.calls["scalars"][0] == (3, {"loss": 1.0})

  def test_empty_image_raises(self, writer):
    values = {"bad_img": np.zeros((0, 0, 0, 3), dtype=np.float32)}
    with pytest.raises(ValueError, match="empty array"):
      metric_writer._write_values_by_type(writer, step=1, values=values)

  def test_empty_values_no_calls(self, writer):
    metric_writer._write_values_by_type(writer, step=1, values={})
    assert not writer.calls


class TestPrepareStepMetrics:

  def test_returns_losses_and_metrics(self):
    aux = mock.MagicMock()
    aux_result = mock.MagicMock()
    aux_result.loss_values = {"losses/total": 1.0}
    aux_result.metric_values = {"metrics/acc": 0.9}
    aux_result.summary_values = {}
    aux.compute.return_value = aux_result

    values = metric_writer.prepare_step_metrics(
        step=10, aux=aux, schedules={}, log_summaries=False
    )
    assert "losses/total" in values
    assert "metrics/acc" in values
    aux.compute.assert_called_once_with(flatten=True)

  def test_includes_summaries_when_requested(self):
    aux = mock.MagicMock()
    aux_result = mock.MagicMock()
    aux_result.loss_values = {}
    aux_result.metric_values = {}
    aux_result.summary_values = {"summaries/img": "placeholder"}
    aux.compute.return_value = aux_result

    values_with = metric_writer.prepare_step_metrics(
        step=1, aux=aux, log_summaries=True
    )
    assert "summaries/img" in values_with

    values_without = metric_writer.prepare_step_metrics(
        step=1, aux=aux, log_summaries=False
    )
    assert "summaries/img" not in values_without

  def test_includes_timer_metrics(self):
    aux = mock.MagicMock()
    aux_result = mock.MagicMock()
    aux_result.loss_values = {}
    aux_result.metric_values = {}
    aux_result.summary_values = {}
    aux.compute.return_value = aux_result

    timer = mock.MagicMock()
    timer.flush_metrics.return_value = {"steps_per_sec": 42.0}

    values = metric_writer.prepare_step_metrics(step=1, aux=aux, timer=timer)
    assert "perf_stats/steps_per_sec" in values
    assert values["perf_stats/steps_per_sec"] == 42.0
