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

"""MLFlowMetricWriter."""

import dataclasses
from typing import Any, Mapping

from etils import epy
from kauldron import konfig
from kauldron.train import metric_writer
from kauldron.typing import Array, Scalar  # pylint: disable=g-multiple-import

with epy.lazy_imports(
    error_callback=(
        "must `pip install 'mlflow>=2.22.0'` to use MLFlowMetricWriter"
    )
):
  import mlflow  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error


@dataclasses.dataclass(frozen=True)
class MLFlowMetricWriter(metric_writer.KDMetricWriter):
  """Simple MLFlow integration for Kauldron Trainer.

  Logs metrics, summaries, config to remote (MLFlow tracking) or local
  MLFlow instance.

  pass_to_default_writer: bool parameter controls whether to pass logging
                               info to default KDMetricWriter

  Example usage in Trainer config:

  ```
  def get_config():
      return kd.train.Trainer(
          ...
          writer=mlflow_metric_writer.MLFlowMetricWriter(pass_to_default_writer=False),
          log_metrics_every = 5,
          log_summaries_every = 1000,
          ...
      )
  ```
  """

  pass_to_default_writer: bool = True

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]) -> None:
    mlflow.log_metrics(
        {
            f"{self.collection}/{scalar_name}": float(value)
            for scalar_name, value in scalars.items()
        },
        step=step,
    )
    if self.pass_to_default_writer:
      return super().write_scalars(step, scalars)

  def write_summaries(
      self,
      step: int,
      values: Mapping[str, Array],
      metadata: Mapping[str, Any] | None = None,
  ) -> None:
    mlflow.log_text(f"{values}\n{metadata}", "summary.txt")
    if self.pass_to_default_writer:
      return super().write_summaries(step, values, metadata)

  def write_config(self, config: konfig.ConfigDict) -> None:
    mlflow.log_text(config.to_json(), "config.json")
    if self.pass_to_default_writer:
      return super().write_config(config)

  def write_element_spec(self, step: int, element_spec) -> None:
    mlflow.log_text(repr(element_spec), "element_spec.txt")
    if self.pass_to_default_writer:
      return super().write_element_spec(step, element_spec)

  def write_param_overview(self, step: int, params) -> None:
    mlflow.log_text(repr(params), "params.txt")
    if self.pass_to_default_writer:
      return super().write_param_overview(step, params)
