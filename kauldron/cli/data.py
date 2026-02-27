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

"""Data-related CLI commands."""

from __future__ import annotations

import dataclasses
from typing import Union

from etils import epy
from kauldron import konfig
from kauldron import kontext
from kauldron.cli import cmd_utils
from kauldron.cli import patch_config


@dataclasses.dataclass(frozen=True, kw_only=True)
class ElementSpec(cmd_utils.SubCommand):
  """Display the element spec of the training data pipeline."""

  # TODO(klausg): add support for eval_ds and other evaluation datasets

  def __call__(
      self,
      cfg: konfig.ConfigDict,
      origin: patch_config.ConfigOrigin | None = None,
  ) -> str:
    trainer = konfig.resolve(cfg)
    elem_spec = trainer.train_ds.element_spec
    # TODO(klausg): What formatting do we want here?
    result = "batch:\n"
    result += "\n".join(
        f"  {k}: {v.dtype}{list(v.shape)}"
        for k, v in kontext.flatten_with_path(elem_spec).items()
    )
    result += "\n"
    return result


@dataclasses.dataclass(frozen=True, kw_only=True)
class Data(cmd_utils.CommandGroup):
  """Data commands."""

  sub_command: Union[ElementSpec]

  def __call__(
      self,
      cfg: konfig.ConfigDict,
      origin: patch_config.ConfigOrigin | None = None,
  ) -> None:
    if origin is not None:
      print(origin.summary())
    epy.pprint(self.sub_command(cfg=cfg, origin=origin))
