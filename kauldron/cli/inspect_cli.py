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

"""Inspect-related CLI commands."""

from __future__ import annotations

import dataclasses
from typing import Union

from kauldron import inspect as kd_inspect
from kauldron.cli import cmd_utils as cu


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelOverview(cu.SubCommand):
  """Display the model overview (parameters, inputs, shapes, etc.)."""

  def __call__(self):
    self.print_config_origin()
    trainer = self.trainer  # trigger config resolution

    with cu.timed("Getting model overview"):
      df = kd_inspect.get_colab_model_overview(
          model=trainer.model,
          train_ds=trainer.train_ds,
          ds_sharding=trainer.sharding.batch,
          model_config=trainer.raw_cfg.model if trainer.raw_cfg else None,
          rngs=trainer.rng_streams.init_rngs(),
      )

    print("\n======== Model Overview ========")
    if hasattr(df, "data"):  # Extract raw DataFrame from StyledDataFrame
      raw_df = df.data
    else:
      raw_df = df

    if hasattr(raw_df, "to_markdown"):
      print(raw_df.to_markdown(index=False))
    else:
      print(raw_df.to_string(index=False))


_SUBCOMMANDS = {
    "model_overview": ModelOverview,
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Inspect(cu.CommandGroup):
  """Inspect commands."""

  sub_command: Union[ModelOverview] = dataclasses.field(
      metadata={"subparsers": _SUBCOMMANDS}
  )
