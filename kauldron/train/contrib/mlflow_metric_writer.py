import logging
import sys
from dataclasses import dataclass
from kauldron.train.metric_writer import KDMetricWriter
from kauldron.typing import Scalar, Array
from typing import Any, Mapping
from kauldron import konfig

try:
    import mlflow
except ImportError:
    logging.error(
        "must `pip install 'mlflow>=2.22.0'` to use MLFlowMetricWriter"
    )
    sys.exit(1)


@dataclass(frozen=True)
class MLFlowMetricWriter(KDMetricWriter):
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
        mlflow.log_text(f"{values}\n{metadata}", f"summary.txt")
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
