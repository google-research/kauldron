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

import abc
import dataclasses
import functools
import json
import sys
from typing import Any, Mapping, Optional

from absl import logging
from clu import metric_writers
from clu import parameter_overview
from etils import epath
from etils import epy
from etils.etree import jax as etree  # pylint: disable=g-importing-member
import jax
from kauldron import konfig
from kauldron import kontext
from kauldron import summaries
from kauldron.data import utils as data_utils
from kauldron.train import auxiliaries
from kauldron.train import trainer_lib
from kauldron.typing import Array, Float, Scalar  # pylint: disable=g-multiple-import
from kauldron.utils import chrono_utils
from kauldron.utils import config_util
from kauldron.utils import constants
from kauldron.utils import kdash
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member
import numpy as np
import optax
import pandas as pd

from unittest import mock as _mock ; xmanager_api = _mock.Mock()

# pylint: disable=logging-fstring-interpolation

COLLECTION_NOT_SET = "$not_set$"


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class WriterBase(abc.ABC, config_util.UpdateFromRootCfg):
  """Base class for metric writers."""

  workdir: str | epath.Path = config_util.ROOT_CFG_REF.workdir

  collection: str = COLLECTION_NOT_SET  # Will be set by the evaluator / trainer

  @abc.abstractmethod
  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]) -> None:
    """Write scalar values for the step."""

  def write_summaries(
      self,
      step: int,
      values: Mapping[str, Array],
      *,
      metadata: Mapping[str, Any] | None = None,
  ) -> None:
    """Write arbitrary tensor summaries for the step."""
    raise NotImplementedError()

  @abc.abstractmethod
  def write_images(
      self,
      step: int,
      images: Mapping[str, Array["n h w c"]],
  ) -> None:
    """Write images for the step."""

  @abc.abstractmethod
  def write_histograms(
      self,
      step: int,
      arrays: Mapping[str, Array],
      *,
      num_buckets: Mapping[str, int] | None = None,
  ) -> None:
    """Write histograms for the step."""

  @abc.abstractmethod
  def write_pointcloud(
      self,
      step: int,
      point_clouds: Mapping[str, Array],
      *,
      point_colors: Optional[Array] = None,
      configs: Optional[Mapping[str, Any]] = None,
  ) -> None:
    """Write point cloud summaries for the step."""

  @abc.abstractmethod
  def write_videos(
      self,
      step: int,
      videos: Mapping[str, Array["n t h w c"]],
  ) -> None:
    """Write videos for the step."""
    raise NotImplementedError()

  @abc.abstractmethod
  def write_audios(
      self,
      step: int,
      audios: Mapping[str, Float["n t c"]],
      *,
      sample_rate: int,
  ) -> None:
    """Write audio samples for the step."""
    raise NotImplementedError()

  @abc.abstractmethod
  def write_texts(
      self,
      step: int,
      texts: Mapping[str, str],
  ) -> None:
    """Write text summaries for the step."""

  @abc.abstractmethod
  def write_hparams(
      self,
      hparams: Mapping[str, Any],
  ) -> None:
    """Write hyper parameters."""

  @abc.abstractmethod
  def write_config(
      self,
      config: konfig.ConfigDict,
  ) -> None:
    """Export the config `.json` file."""

  @abc.abstractmethod
  def write_param_overview(self, step: int, params) -> None:
    """Write a table with parameter shapes and sizes."""

  @abc.abstractmethod
  def write_element_spec(self, step: int, element_spec) -> None:
    """Write the element spec."""

  @abc.abstractmethod
  def write_context_structure(
      self, step: int, trainer: trainer_lib.Trainer
  ) -> None:
    """Write the context structure."""

  # TODO(b/378060021): tidy this function up after old summaries are removed
  def write_step_metrics(
      self,
      *,
      step: int,
      aux: auxiliaries.AuxiliariesState,
      schedules: Mapping[str, optax.Schedule],
      log_summaries: bool,
      timer: Optional[chrono_utils.Chrono] = None,
  ) -> None:
    """Logs scalar and image summaries."""
    aux_result = aux.compute(flatten=True)

    # schedules
    schedule_values = jax.tree.map(
        lambda s: _compute_schedule(s, step), schedules
    )
    schedule_values = kontext.flatten_with_path(
        schedule_values, prefix="schedules", separator="/"
    )

    if timer:
      performance_stats = {
          f"perf_stats/{k}": v
          for k, v in timer.flush_metrics().items()
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
        image_summaries = {
            name: value
            for name, value in aux_result.summary_values.items()
            if isinstance(value, Float["n h w #3"])
        }
      # Throw an error if empty arrays are given. TB throws very odd errors
      # and kills Colab runtimes if we don't catch these ourselves.
      for name, image in image_summaries.items():
        if image.size == 0:
          raise ValueError(
              f"Image summary `{name}` is empty array of shape {image.shape}."
          )
      self.write_images(step=step, images=image_summaries)

      with jax.spmd_mode("allow_all"), jax.transfer_guard("allow"):
        # histograms
        hist_summaries = {
            name: value
            for name, value in aux_result.summary_values.items()
            if isinstance(value, summaries.Histogram)
        }

        self.write_histograms(
            step=step,
            arrays={k: hist.tensor for k, hist in hist_summaries.items()},
            num_buckets={
                k: hist.num_buckets for k, hist in hist_summaries.items()
            },
        )

      with jax.spmd_mode("allow_all"), jax.transfer_guard("allow"):
        # point clouds
        pc_summaries = {
            name: value
            for name, value in aux_result.summary_values.items()
            if isinstance(value, summaries.PointCloud)
        }
        self.write_pointcloud(
            step=step,
            point_clouds={
                k: point_cloud.point_clouds
                for k, point_cloud in pc_summaries.items()
            },
            point_colors={
                k: point_cloud.point_colors
                for k, point_cloud in pc_summaries.items()
            },
            configs={
                k: point_cloud.configs
                for k, point_cloud in pc_summaries.items()
            },
        )

    # TODO(epot): This is blocking and slow. Is it really required ?
    # Should likely be only called once at the end of the training / eval.
    self.flush()

  def flush(self) -> None:
    pass

  def close(self) -> None:
    pass

  def _assert_collection_is_set(self) -> None:
    if self.collection is COLLECTION_NOT_SET:
      raise ValueError("collection name must be set.")

  replace = dataclasses.replace


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class MetadataWriter(WriterBase):
  """Mixing to log global metadata."""

  @abc.abstractmethod
  def write_hparams(
      self,
      hparams: Mapping[str, Any],
  ) -> None:
    """Write hyper parameters."""
    # TODO(epot): Could have implementation with `self.write_texts(0, texts)`

  def write_config(
      self,
      config: konfig.ConfigDict,
  ) -> None:
    self._assert_collection_is_set()
    if config is None:
      return

    if status.is_lead_host:
      # Save the raw config (for easy re-loading)
      config_path = self.workdir / constants.CONFIG_FILENAME
      config_path.write_text(config.to_json())

    texts = {"config": f"```python\n{config!r}\n```"}
    self.write_texts(0, texts)

  def write_param_overview(self, step: int, params) -> None:
    self._assert_collection_is_set()
    texts = {"parameters": _get_markdown_param_table(params)}
    self.write_texts(step, texts)

  def write_element_spec(self, step: int, element_spec) -> None:
    self._assert_collection_is_set()

    if status.is_lead_host:
      # Save the raw config (for easy re-loading)
      spec_path = self.workdir / constants.ELEMENT_SPEC_FILENAME
      element_spec = data_utils.spec_to_json(element_spec)
      element_spec = json.dumps(element_spec, indent=2)
      spec_path.write_text(element_spec)

    texts = {"element_spec": f"```python\n{element_spec!s}\n```"}
    self.write_texts(step, texts)


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class NoopMetadataWriter(WriterBase):
  """Mixing to disable loggin global metadata."""

  def write_hparams(self, hparams: Mapping[str, Any]) -> None:
    pass

  def write_config(
      self,
      config: konfig.ConfigDict,
  ) -> None:
    pass

  def write_param_overview(self, step: int, params) -> None:
    pass

  def write_element_spec(self, step: int, element_spec) -> None:
    pass

  def write_context_structure(
      self, step: int, trainer: trainer_lib.Trainer
  ) -> None:
    pass


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class KDMetricWriter(MetadataWriter):
  """Writes summaries to logs, tf_summaries and datatables.

  Differs from the clu default metric writer in a few ways:
   - It divides summaries into two datatables: one for scalars and one for
     arrays to improve datatable access speed for flatboards.
   - Doesn't write hyperparameters to the datatable to avoid clutter.
   - Does not write to XM-Measurements.
   - offers additional methods to write config, param_overview and element_spec
  """

  add_artifacts: bool = True

  @functools.cached_property
  def _collection_path_prefix(self) -> str:
    return ""

  @functools.cached_property
  def _scalar_datatable_name(self) -> str:
    self._assert_collection_is_set()
    return f"{self._collection_path_prefix}{self.collection}"

  @functools.cached_property
  def _array_datatable_name(self) -> str:
    self._assert_collection_is_set()
    return f"{self._collection_path_prefix}{self.collection}_arrays"

  @functools.cached_property
  def _noop(self) -> metric_writers.MetricWriter:
    return metric_writers.MultiWriter([])

  @functools.cached_property
  def _log_writer(self) -> metric_writers.MetricWriter:
    self._assert_collection_is_set()
    return metric_writers.AsyncWriter(
        metric_writers.LoggingWriter(self.collection)
    )

  @functools.cached_property
  def _tf_summary_writer(self) -> metric_writers.MetricWriter:
    if status.is_lead_host:
      self._assert_collection_is_set()
      return metric_writers.SummaryWriter(
          logdir=str(self.workdir / self.collection)
      )
    else:
      return self._noop

  @functools.cached_property
  def _scalar_writer(self) -> metric_writers.MetricWriter:
    return self._create_datatable_writer(
        name=self._scalar_datatable_name,
        description=f"Scalars datatable ({self.collection})",
    )

  @functools.cached_property
  def _array_writer(self) -> metric_writers.MetricWriter:
    return self._create_datatable_writer(
        name=self._array_datatable_name,
        description=f"Arrays and images datatable ({self.collection})",
    )

  def _create_datatable_writer(
      self, name: str, description: str
  ) -> metric_writers.MetricWriter:
    return self._noop

  def write_summaries(
      self,
      step: int,
      values: Mapping[str, Array],
      metadata: Mapping[str, Any] | None = None,
  ) -> None:
    self._array_writer.write_summaries(step, values, metadata)
    self._tf_summary_writer.write_summaries(step, values, metadata)

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]) -> None:
    self._log_writer.write_scalars(step, scalars)
    self._scalar_writer.write_scalars(step, scalars)
    self._tf_summary_writer.write_scalars(step, scalars)

  def write_images(
      self, step: int, images: Mapping[str, Array["n h w c"]]
  ) -> None:
    images_uint8 = {}
    for key, image in images.items():
      if isinstance(image, Float["n h w c"]):
        # DatatableUI autoscales float images, so convert to uint8
        image = np.array(np.clip(image * 255.0, 0.0, 255.0), dtype=np.uint8)
      images_uint8[key] = image

    self._array_writer.write_images(step, images_uint8)
    self._tf_summary_writer.write_images(step, images_uint8)

  def write_histograms(
      self,
      step: int,
      arrays: Mapping[str, Array],
      num_buckets: Mapping[str, int] | None = None,
  ) -> None:
    self._tf_summary_writer.write_histograms(step, arrays, num_buckets)

  def write_videos(
      self, step: int, videos: Mapping[str, Array["n t h w c"]]
  ) -> None:
    self._tf_summary_writer.write_videos(step, videos)

  def write_audios(
      self,
      step: int,
      audios: Mapping[str, Float["n t c"]],
      *,
      sample_rate: int,
  ) -> None:
    self._tf_summary_writer.write_audios(step, audios, sample_rate=sample_rate)

  def write_texts(self, step: int, texts: Mapping[str, str]) -> None:
    # log_writer.write_texts does not display new lines, making it hard to
    # parse the text.
    logging.info("Writing texts:")
    for k, v in texts.items():
      logging.info(f"{k}: {v}")
    self._log_writer.write_texts(step, texts)
    self._tf_summary_writer.write_texts(step, texts)

  def write_pointcloud(
      self,
      step: int,
      point_clouds: Mapping[str, Array["n 3"]],
      *,
      point_colors: Mapping[str, Array["n 3"]] | None = None,
      configs: Mapping[str, str | float | bool | None] | None = None,
  ) -> None:
    if not point_clouds:
      return
    logging.info("Pointcloud summary not supported.")

  def write_hparams(self, hparams: Mapping[str, Any]) -> None:
    self._log_writer.write_hparams(hparams)
    self._tf_summary_writer.write_hparams(hparams)

  def write_context_structure(
      self, step: int, trainer: trainer_lib.Trainer
  ) -> None:
    self._assert_collection_is_set()
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
        [(f"`{k}`", f"`{v}`") for k, v in context_spec.items()],  # pylint: disable=attribute-error
        columns=["Path", "Spec"],
    )
    # export pandas dataframe as markdown text
    markdown_table = ctx_df.to_markdown(index=False, tablefmt="github")
    self.write_texts(step, {"context_spec": markdown_table})

  def flush(self) -> None:
    self._scalar_writer.flush()
    self._array_writer.flush()
    self._tf_summary_writer.flush()

  def close(self) -> None:
    self._scalar_writer.close()
    self._array_writer.close()
    self._tf_summary_writer.close()


@dataclasses.dataclass(frozen=True, eq=True, kw_only=True)
class NoopWriter(NoopMetadataWriter):
  """Writer that writes nothing. Useful for deactivating the writer."""

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]) -> None:
    pass

  def write_summaries(
      self,
      step: int,
      values: Mapping[str, Array],
      metadata: Mapping[str, Any] | None = None,
  ) -> None:
    pass

  def write_images(
      self, step: int, images: Mapping[str, Array["n h w c"]]
  ) -> None:
    pass

  def write_histograms(
      self,
      step: int,
      arrays: Mapping[str, Array],
      num_buckets: Mapping[str, int] | None = None,
  ) -> None:
    pass

  def write_pointcloud(
      self,
      step: int,
      point_clouds: Array["n 3"],
      *,
      point_colors: Array["n 3"] | None = None,
      configs: Mapping[str, str | float | bool | None] | None = None,
  ) -> None:
    pass

  def write_videos(
      self, step: int, videos: Mapping[str, Array["n t h w c"]]
  ) -> None:
    pass

  def write_audios(
      self,
      step: int,
      audios: Mapping[str, Float["n t c"]],
      *,
      sample_rate: int,
  ) -> None:
    pass

  def write_texts(self, step: int, texts: Mapping[str, str]) -> None:
    pass

  def write_step_metrics(
      self,
      *,
      step: int,
      aux: auxiliaries.AuxiliariesState,
      schedules: Mapping[str, optax.Schedule],
      log_summaries: bool,
      timer: Optional[chrono_utils.Chrono] = None,
  ) -> None:
    pass


def _get_markdown_param_table(params) -> str:
  """Returns a markdown table of the parameter overview."""
  param_table = parameter_overview.get_parameter_overview(
      params, include_stats="global"
  )
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
