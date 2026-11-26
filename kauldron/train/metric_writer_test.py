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

import threading

from clu import metric_writers
from etils import epath
from kauldron.train import metric_writer
import numpy as np


class _FakeWriter(metric_writers.MultiWriter):
  """Records writes and flushes. `flush()` blocks until `release` is set."""

  def __init__(
      self, *, block_flush: bool = True, fail_with=None, fail_on=None
  ):
    super().__init__([])
    self.calls = []
    self._block_flush = block_flush
    self._fail_with = fail_with
    self._fail_on = fail_on
    self.flush_started = threading.Event()
    self.release = threading.Event()

  def _maybe_fail(self, op: str):
    if self._fail_with is not None and (
        self._fail_on is None or op in self._fail_on
    ):
      raise self._fail_with

  def write_scalars(self, step, scalars):
    del scalars
    self.calls.append(f'write_{step}')
    self._maybe_fail('write_scalars')

  def write_images(self, step, images):
    del images
    self.calls.append(f'write_images_{step}')
    self._maybe_fail('write_images')

  def write_summaries(self, step, values, metadata=None):
    del values, metadata
    self.calls.append(f'write_summaries_{step}')
    self._maybe_fail('write_summaries')

  def flush(self):
    self.calls.append('flush')
    self.flush_started.set()
    if self._block_flush:
      assert self.release.wait(timeout=30)
    self._maybe_fail('flush')

  def close(self):
    self.calls.append('close')
    self._maybe_fail('close')


def _make_kd_writer(
    tmp_path,
    *,
    ignore_datatable_errors: bool = False,
) -> metric_writer.KDMetricWriter:
  kwargs = {}
  return metric_writer.KDMetricWriter(
      workdir=epath.Path(tmp_path),
      collection='train',
      ignore_datatable_errors=ignore_datatable_errors,
      **kwargs,
  )


def test_flush_in_background_does_not_wait():
  fake = _FakeWriter()
  writer = metric_writer._BackgroundFlushAsyncWriter(fake)  # pylint: disable=protected-access
  writer.write_scalars(0, {'loss': 1.0})
  writer.flush_in_background()  # Returns while the flush is still blocked.
  assert fake.flush_started.wait(timeout=30)
  writer.write_scalars(1, {'loss': 0.5})
  fake.release.set()
  writer.flush()
  assert fake.calls == ['write_0', 'flush', 'write_1', 'flush']


def test_background_flushes_do_not_pile_up():
  fake = _FakeWriter()
  writer = metric_writer._BackgroundFlushAsyncWriter(fake)  # pylint: disable=protected-access
  writer.flush_in_background()
  assert fake.flush_started.wait(timeout=30)  # The first flush is running.
  for _ in range(5):
    writer.flush_in_background()  # Only the last of these flushes.
  fake.release.set()
  writer.flush()
  # Running flush + the last background flush + the final blocking flush.
  assert fake.calls.count('flush') == 3


def test_background_flush_covers_writes_logged_while_flushing():
  fake = _FakeWriter()
  writer = metric_writer._BackgroundFlushAsyncWriter(fake)  # pylint: disable=protected-access
  writer.write_scalars(0, {'loss': 1.0})
  writer.flush_in_background()
  assert fake.flush_started.wait(timeout=30)  # The first flush is running.
  # Logging is faster than flushing: steps 1-3 are logged meanwhile.
  writer.write_scalars(1, {'loss': 0.9})
  writer.flush_in_background()
  writer.write_scalars(2, {'loss': 0.8})
  writer.flush_in_background()
  writer.write_scalars(3, {'loss': 0.7})
  writer.flush_in_background()
  fake.release.set()
  # Wait for the writer thread to run everything queued.
  writer._pool.join()  # pylint: disable=protected-access
  # The last write is followed by a flush; the flushes before it are skipped.
  assert fake.calls == [
      'write_0', 'flush', 'write_1', 'write_2', 'write_3', 'flush'
  ]


def test_empty_writes_and_lifecycle_do_not_create_summary_dir(tmp_path):
  writer = _make_kd_writer(tmp_path)
  writer.write_scalars(0, {})
  writer.write_images(0, {})
  writer.write_summaries(0, {})
  writer.write_histograms(0, {})
  writer.write_videos(0, {})
  writer.write_audios(0, {}, sample_rate=16000)
  writer.write_texts(0, {})
  writer.write_hparams({})
  writer.flush_in_background()
  writer.wait_for_background_flush()
  writer.flush()
  writer.close()
  assert not (tmp_path / 'train').exists()


def test_nonempty_image_write_creates_summary_dir(tmp_path):
  writer = _make_kd_writer(tmp_path)
  writer.write_images(0, {'img': np.zeros((1, 4, 4, 3), dtype=np.uint8)})
  writer.flush()
  writer.close()
  assert (tmp_path / 'train').exists()
