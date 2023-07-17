# Copyright 2023 The kauldron Authors.
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

"""Custom MetricWriter."""

from __future__ import annotations

from typing import Any, Mapping

from clu import metric_writers
from etils import epath
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member
from kauldron.typing import Array, Float, Scalar  # pylint: disable=g-multiple-import


class KDMetricWriter(metric_writers.MetricWriter):
  """Writes summaries to logs, tf_summaries and datatables.

  Differs from the clu default metric writer in a few ways:
   - It divides summaries into two datatables: one for scalars and one for
     arrays to improve datatable access speed for flatboards.
   - Doesn't write hyperparameters to the datatable to avoid clutter.
   - Does not write to XM-Measurements.
  """

  def __init__(self, workdir: epath.PathLike, collection: str):
    self.workdir = epath.Path(workdir)
    self.collection = collection
    self.just_logging = not status.on_xmanager or not status.is_lead_host
    if self.just_logging:
      self.scalar_writer = metric_writers.AsyncWriter(
          metric_writers.LoggingWriter(collection)
      )
      self.array_writer = metric_writers.MultiWriter([])
      self.tf_summary_writer = metric_writers.MultiWriter([])
    else:
      self.scalar_writer = metric_writers.AsyncMultiWriter([
          metric_writers.LoggingWriter(collection),
          metric_writers.DatatableWriter(
              datatable_name=f"/datatable/xid/{status.xid}/{collection}",
              keys=[("wid", status.wid)],
          ),
      ])
      self.array_writer = metric_writers.AsyncWriter(
          metric_writers.DatatableWriter(
              datatable_name=f"/datatable/xid/{status.xid}/arrays",
              keys=[("wid", status.wid), ("collection", collection)],
          ),
      )
      self.tf_summary_writer = metric_writers.SummaryWriter(
          logdir=str(self.workdir / collection)
      )

  def write_summaries(
      self,
      step: int,
      values: Mapping[str, Array],
      metadata: Mapping[str, Any] | None = None,
  ):
    self.array_writer.write_summaries(step, values, metadata)
    self.tf_summary_writer.write_summaries(step, values, metadata)

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    self.scalar_writer.write_scalars(step, scalars)
    self.tf_summary_writer.write_scalars(step, scalars)

  def write_images(self, step: int, images: Mapping[str, Array["N H W C"]]):
    self.array_writer.write_images(step, images)
    self.tf_summary_writer.write_images(step, images)

  def write_histograms(
      self,
      step: int,
      arrays: Mapping[str, Array],
      num_buckets: Mapping[str, int] | None = None,
  ):
    self.tf_summary_writer.write_histograms(step, arrays, num_buckets)

  def write_videos(self, step: int, videos: Mapping[str, Array["N T H W C"]]):
    self.tf_summary_writer.write_videos(step, videos)

  def write_audios(
      self,
      step: int,
      audios: Mapping[str, Float["N T C"]],
      *,
      sample_rate: int,
  ):
    self.tf_summary_writer.write_audios(step, audios, sample_rate=sample_rate)

  def write_texts(self, step: int, texts: Mapping[str, str]):
    self.tf_summary_writer.write_texts(step, texts)

  def write_hparams(self, hparams: Mapping[str, Any]):
    self.tf_summary_writer.write_hparams(hparams)

  def flush(self):
    self.scalar_writer.flush()
    self.array_writer.flush()
    self.tf_summary_writer.flush()

  def close(self):
    self.scalar_writer.close()
    self.array_writer.close()
    self.tf_summary_writer.close()
