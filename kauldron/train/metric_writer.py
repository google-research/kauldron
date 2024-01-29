# Copyright 2024 The kauldron Authors.
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

from typing import Any, Mapping, Optional

from clu import metric_writers
from clu import parameter_overview
from etils import epath
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import jax
from kauldron import konfig
from kauldron import kontext
from kauldron import summaries
from kauldron.train import config_lib
from kauldron.train import timer as timer_module
from kauldron.train import train_step
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member
from kauldron.typing import Array, Float, Scalar  # pylint: disable=g-multiple-import
import numpy as np
import optax
import pandas as pd

from unittest import mock as _mock ; xmanager_api = _mock.Mock()


class KDMetricWriter(metric_writers.MetricWriter):
  """Writes summaries to logs, tf_summaries and datatables.

  Differs from the clu default metric writer in a few ways:
   - It divides summaries into two datatables: one for scalars and one for
     arrays to improve datatable access speed for flatboards.
   - Doesn't write hyperparameters to the datatable to avoid clutter.
   - Does not write to XM-Measurements.
   - offers additional methods to write config, param_overview and element_spec
  """

  def __init__(self, workdir: epath.PathLike, collection: str):
    self.workdir = epath.Path(workdir)
    self.collection = collection
    self.log_writer = metric_writers.AsyncWriter(
        metric_writers.LoggingWriter(collection)
    )
    noop = metric_writers.MultiWriter([])
    if status.is_lead_host:
      self.tf_summary_writer = metric_writers.SummaryWriter(
          logdir=str(self.workdir / collection)
      )
    else:
      self.tf_summary_writer = noop

    if status.on_xmanager and status.is_lead_host:
      self.scalar_writer = metric_writers.AsyncWriter(
          metric_writers.DatatableWriter(
              datatable_name=self.scalar_datatable_name,
              keys=[("wid", status.wid)],
          ),
      )
      self.array_writer = metric_writers.AsyncWriter(
          metric_writers.DatatableWriter(
              datatable_name=self.array_datatable_name,
              keys=[("wid", status.wid)],
          ),
      )
    else:
      self.scalar_writer = noop
      self.array_writer = noop

    self.add_artifacts()

  @property
  def scalar_datatable_name(self) -> str:
    if not status.on_xmanager:
      raise RuntimeError("Not on XManager.")
    return f"/datatable/xid/{status.xid}/{self.collection}"

  @property
  def array_datatable_name(self) -> str:
    if not status.on_xmanager:
      raise RuntimeError("Not on XManager.")
    return f"/datatable/xid/{status.xid}/{self.collection}_arrays"

  def write_summaries(
      self,
      step: int,
      values: Mapping[str, Array],
      metadata: Mapping[str, Any] | None = None,
  ):
    self.array_writer.write_summaries(step, values, metadata)
    self.tf_summary_writer.write_summaries(step, values, metadata)

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    self.log_writer.write_scalars(step, scalars)
    self.scalar_writer.write_scalars(step, scalars)
    self.tf_summary_writer.write_scalars(step, scalars)

  def write_images(self, step: int, images: Mapping[str, Array["N H W C"]]):
    images_uint8 = {}
    for key, image in images.items():
      if isinstance(image, Float["N H W C"]):
        # DatatableUI autoscales float images, so convert to uint8
        image = np.array(np.clip(image * 255.0, 0.0, 255.0), dtype=np.uint8)
      images_uint8[key] = image

    self.array_writer.write_images(step, images_uint8)
    self.tf_summary_writer.write_images(step, images_uint8)

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
    self.log_writer.write_texts(step, texts)
    self.tf_summary_writer.write_texts(step, texts)

  def write_hparams(self, hparams: Mapping[str, Any]):
    self.log_writer.write_hparams(hparams)
    self.tf_summary_writer.write_hparams(hparams)

  def write_config(self, config: konfig.ConfigDict) -> None:
    if config is None:
      return

    if status.is_lead_host:
      # Save the raw config (for easy re-loading)
      config_path = self.workdir / "config.json"
      config_path.write_text(config.to_json())

    texts = {"config": f"```python\n{config!r}\n```"}
    self.write_texts(0, texts)

  def write_param_overview(self, step: int, params):
    texts = {"parameters": get_markdown_param_table(params)}
    self.write_texts(step, texts)

  def write_element_spec(self, step: int, element_spec):
    texts = {"element_spec": f"```python\n{element_spec!s}\n```"}
    self.write_texts(step, texts)

  def write_context_structure(
      self, step: int, trainer: config_lib.Trainer
  ) -> None:
    # do a lightweight shape-eval for the context
    context = trainer.context_specs
    # create a flat spec for the context
    context_spec = kontext.flatten_with_path(context)
    context_spec = etree.spec_like(context_spec)
    context_spec["grads"] = "<<same structure as params>>"
    context_spec["updates"] = "<<same structure as params>>"

    # convert flat spec into a pandas dataframe
    ctx_df = pd.DataFrame(
        # wrap entries in backticks to avoid interpreting __x__ as markdown bold
        [(f"`{k}`", f"`{v}`") for k, v in context_spec.items()],
        columns=["Path", "Spec"],
    )
    # export pandas dataframe as markdown text
    markdown_table = ctx_df.to_markdown(index=False, tablefmt="github")
    self.write_texts(step, {"context_spec": markdown_table})

  def write_step_metrics(
      self,
      *,
      step: int,
      aux: train_step.Auxiliaries,
      model_with_aux: train_step.ModelWithAux,
      schedules: Mapping[str, optax.Schedule],
      log_summaries: bool,
      timer: Optional[timer_module.PerformanceTimer] = None,
  ):
    """Logs scalar and image summaries."""
    aux_result = aux.compute(flatten=True)

    # schedules
    schedule_values = jax.tree_map(
        lambda s: _compute_schedule(s, step), schedules
    )
    schedule_values = kontext.flatten_with_path(
        schedule_values, prefix="schedules", separator="/"
    )

    if timer:
      performance_stats = {
          f"perf_stats/{k}": v
          for k, v in timer.log_stats(step_num=step).items()
      }
    else:
      performance_stats = {}
    with jax.transfer_guard("allow"):
      self.write_scalars(
          step=step,
          scalars=(
              aux_result.loss_values
              | aux_result.metric_values
              | schedule_values
              | performance_stats
          ),
      )

    if log_summaries:
      with jax.transfer_guard("allow"):
        # image summaries  # TODO(klausg): unify with metrics
        image_summaries = {
            name: summary.get_images(**aux.summary_kwargs[name])
            for name, summary in model_with_aux.summaries.items()
            if isinstance(summary, summaries.ImageSummary)
        }
      # Throw an error if empty arrays are given. TB throws very odd errors
      # and kills Colab runtimes if we don't catch these ourselves.
      for name, image in image_summaries.items():
        if image.size == 0:
          raise ValueError(
              f"Image summary `{name}` is empty array of shape {image.shape}."
          )
      self.write_images(step=step, images=image_summaries)

      # histograms
      hist_summaries = {
          name: summary.get_tensor(**aux.summary_kwargs[name])
          for name, summary in model_with_aux.summaries.items()
          if isinstance(summary, summaries.HistogramSummary)
      }
      for name, (_, tensor) in hist_summaries.items():
        if tensor.size == 0:
          raise ValueError(
              f"Histogram summary `{name}` is empty array of shape"
              f" {tensor.shape}."
          )
      self.write_histograms(
          step=step,
          arrays={k: tensor for k, (_, tensor) in hist_summaries.items()},
          num_buckets={
              k: n_buckets for k, (n_buckets, _) in hist_summaries.items()
          },
      )

    self.flush()

  def add_artifacts(self):
    if not (status.on_xmanager and status.is_lead_host and status.wid == 1):
      return  # only add artifacts from lead host of first work unit on XM

    status.xp.create_artifact(
        artifact_type=xmanager_api.ArtifactType.ARTIFACT_TYPE_STORAGE2_BIGTABLE,
        artifact=self.array_datatable_name,
        description=f"Arrays and images datatable ({self.collection})",
    )
    status.xp.create_artifact(
        artifact_type=xmanager_api.ArtifactType.ARTIFACT_TYPE_STORAGE2_BIGTABLE,
        artifact=self.scalar_datatable_name,
        description=f"Scalars datatable ({self.collection})",
    )

  def flush(self):
    self.scalar_writer.flush()
    self.array_writer.flush()
    self.tf_summary_writer.flush()

  def close(self):
    self.scalar_writer.close()
    self.array_writer.close()
    self.tf_summary_writer.close()


def get_markdown_param_table(params):
  param_table = parameter_overview.get_parameter_overview(params)
  # convert to markdown format (Only minor adjustments needed)
  rows = param_table.split("\n")
  header = rows[1]
  hline = rows[2].replace("+", "|")  # markdown syntax
  body = rows[3:-2]
  total = rows[-1]
  return "\n".join([header, hline] + body + ["", total])


def _compute_schedule(sched: optax.Schedule, step: int):
  """Evaluate schedule for step and return result."""
  with jax.transfer_guard("allow"):
    return sched(step)
