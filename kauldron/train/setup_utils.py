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
import typing

from absl import logging
from etils import epath
import tensorflow as tf

# Do not import `trainer_lib` at runtime to avoid circular imports
if typing.TYPE_CHECKING:
  from kauldron.train import trainer_lib  # pylint: disable=g-bad-import-order


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
    add_flatboard: Whether to create the flatboard dashboards.
    flatboard_build_context: Shared info to build the flatboard dashboards. This
      object is created once globally and shared between all the metrics
      `Writer`, to ensure all dashboards are written to the same collection.
    eval_only: Whether the job is a eval-only job.
  """

  # Could provide more options here to customize artifacts,...

  tags: str | list[str] = dataclasses.field(default_factory=list)
  tqdm_info: TqdmInfo = dataclasses.field(default_factory=TqdmInfo)
  eval_only: bool = False

  def __post_init__(self):
    # Normalize tags to a list.
    if isinstance(self.tags, str):
      object.__setattr__(self, "tags", self.tags.split(","))

  def run(self, trainer: trainer_lib.Trainer) -> None:
    """Perform the initial setup."""
    tf.config.set_visible_devices([], "GPU")

    _create_workdir(trainer.workdir)

  def log_status(self, msg: str) -> None:
    logging.info(msg)


def _create_workdir(workdir: epath.PathLike):
  """Ensure workdir is set and exists."""
  workdir = epath.Path(workdir) if workdir else epath.Path()
  if workdir == epath.Path():
    raise ValueError("--workdir must be set when running on XManager.")

  logging.info("Creating workdir: %s", workdir)
  workdir.mkdir(parents=True, exist_ok=True)
