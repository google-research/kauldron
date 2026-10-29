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
from kauldron.train import metric_writer


class _FakeWriter(metric_writers.MultiWriter):
  """Records writes and flushes. `flush()` blocks until `release` is set."""

  def __init__(self):
    super().__init__([])
    self.calls = []
    self.flush_started = threading.Event()
    self.release = threading.Event()

  def write_scalars(self, step, scalars):
    del scalars
    self.calls.append(f'write_{step}')

  def flush(self):
    self.calls.append('flush')
    self.flush_started.set()
    assert self.release.wait(timeout=30)


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
