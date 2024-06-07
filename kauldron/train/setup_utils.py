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

"""Dataclasses."""

from __future__ import annotations

import dataclasses

from absl import logging
from etils import epath
from etils import exm
from kauldron.train import config_lib
from kauldron.train import flatboard_utils
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member
from kauldron.utils import utils
import tensorflow as tf


@dataclasses.dataclass(kw_only=True, frozen=True)
class TqdmInfo:
  desc: str = "train"
  log_xm: bool = True


@dataclasses.dataclass(kw_only=True, frozen=True)
class Setup:
  """Setup/environment options.

  Attributes:
    tags: Custom XManager tags.
    tqdm_info: Customize the `tqdm` bar.
  """

  # Could provide more options here to customize artifacts,...

  tags: str | list[str] = dataclasses.field(default_factory=list)
  tqdm_info: TqdmInfo = dataclasses.field(default_factory=TqdmInfo)

  def __post_init__(self):
    # Normalize tags to a list.
    if isinstance(self.tags, str):
      object.__setattr__(self, "tags", self.tags.split(","))

  def run(self, trainer: config_lib.Trainer) -> None:
    tf.config.set_visible_devices([], "GPU")

    utils.add_log_artifacts()
    utils.add_colab_artifacts()
    flatboard_utils.add_flatboards(trainer)
    _ensure_workdir(trainer.workdir)

    if self.tags and status.is_lead_host:
      assert isinstance(self.tags, list)
      experiment = exm.current_experiment()
      experiment.add_tags(*self.tags)

  def log(self, msg: str) -> None:
    status.log(msg)


def _ensure_workdir(workdir: epath.PathLike):
  """Ensure workdir is set and exists."""
  workdir = epath.Path(workdir) if workdir else epath.Path()
  if workdir == epath.Path():
    raise ValueError("--workdir must be set when running on XManager.")

  logging.info("Creating workdir: %s", workdir)
  workdir.mkdir(parents=True, exist_ok=True)
