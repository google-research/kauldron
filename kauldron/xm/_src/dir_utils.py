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

"""WorkdirInfo."""

from __future__ import annotations

import dataclasses
import functools
import os
import textwrap
from typing import Any

from etils import epy
from etils import exm
from kauldron.xm._src import job_lib
from kauldron.xm._src import sweep_utils
from xmanager import xm_abc


_MISSING: Any = object()


@dataclasses.dataclass(frozen=True, kw_only=True)
class DirContext:
  """Context used for formatting."""

  # After experiment is created
  xp: xm_abc.XManagerExperiment = _MISSING
  resolved_jobs: dict[str, job_lib.Job] = _MISSING
  sweep_info: sweep_utils.SweepInfo = _MISSING

  # After the work-unit is created
  wu: xm_abc.XManagerWorkUnit = _MISSING
  sweep_item: sweep_utils.SweepItem = _MISSING

  # When set to `True`, calling `ctx.xp` will raise error if `xp` is not set.
  _ensure_not_missing: bool = False

  def __getattribute__(self, name: str):
    value = object.__getattribute__(self, name)
    if (
        object.__getattribute__(self, "_ensure_not_missing")
        and value is _MISSING
    ):
      raise ValueError(f"{name} not set yet.")
    return value


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubdirFormat:
  """Specify formatting options for the workdir for an experiment/work-unit.

  Usage:

  ```python
  xp = kxm.Experiment(
      subdir_format=kxm.SubdirFormat(xp_dirname='{xid}-{name}')
  )
  ```

  Subclasses can define add `def <my_attributes>(self, ctx):` to support
  additional `{my_attributes}`.

  For example, to create a new `{creation_date}` option:

  ```
  class MyCustomFormat(SubdirFormat):

    def creation_date(self, ctx) -> str:
      return _get_and_format_date(ctx.xp)
  ```

  Attributes:
    xp_dirname: Format of the experiment directory
    wu_dirname: Format name of the experiment directory
  """

  xp_dirname: str = "{xid}"
  wu_dirname: str = "{wid}"

  def __post_init__(self):
    # Normalize dirname to support `--xp.subdir_format.xp_dirname=12345`
    object.__setattr__(self, "xp_dirname", str(self.xp_dirname))
    object.__setattr__(self, "wu_dirname", str(self.wu_dirname))

  # Available at experiment creation

  def xid(self, ctx: DirContext) -> int:
    """Experiment id."""
    return ctx.xp.experiment_id

  def name(self, ctx: DirContext) -> str:
    """Name of the experiment."""
    return ctx.xp.context.annotations.title

  def title(self, ctx: DirContext) -> str:
    """Alias of `{name}`."""
    return self.name(ctx)

  def author(self, ctx: DirContext) -> str:
    """Author of the experiment."""
    return ctx.xp.context.creator

  def user(self, ctx: DirContext) -> str:
    """Alias for `{author}`."""
    return self.author(ctx)

  def cell(self, ctx: DirContext) -> str:
    """Cell (`jn`, `lu`,...)."""
    # TODO(epot): Better mechanism to select the main cell

    # If the work-unit are launched on different directories, they will
    # still use the same cell from the xp_workdir. Might be desirable (e.g.
    # TB log from a single place)
    main_job = list(ctx.resolved_jobs.values())[0]
    assert main_job.cell
    return main_job.cell

  # Available at work-unit creation

  def wid(self, ctx: DirContext) -> str:
    """Work unit id (0-padded to the number of work-units: `001`,...)."""
    num_sweeps = len(ctx.sweep_info)
    num_digits = len(str(num_sweeps))
    return f"{ctx.wu.work_unit_id:0{num_digits}}"

  def unpadded_wid(self, ctx: DirContext) -> int:
    """Work unit id (unpadded)."""
    return ctx.wu.work_unit_id

  def separator_if_sweep(self, ctx: DirContext) -> str:
    """`-` only if there's a sweep."""
    # Use `xm_ui_kwargs` as likely closer to what human-readable should be
    return "-" if ctx.sweep_item.job_kwargs else ""

  def sweep_kwargs(self, ctx: DirContext) -> str:
    """Format the sweep."""
    # Use `xm_ui_kwargs` as likely closer to what human-readable should be
    return _format_sweep_kwargs(ctx.sweep_item.xm_ui_kwargs)


def _format_sweep_kwargs(sweep_kwargs: dict[str, Any]) -> str:
  parts = []
  for k, v in sweep_kwargs.items():
    # TODO(epot): Should also shorten the key/values name for long arguments
    parts.append(f"{k}={v}")
  sweep_str = ",".join(parts)
  sweep_str = _filter_invalid_chars(sweep_str)
  sweep_str = textwrap.shorten(sweep_str, width=127, placeholder="...")
  return sweep_str


def _filter_invalid_chars(sweep_str: str) -> str:
  sweep_str = sweep_str.replace("[", "(")
  sweep_str = sweep_str.replace("]", ")")
  # The following chars seems to work: # { ( : ' " ,
  for c in ("*", "\n", "%", "/", "\\"):
    sweep_str = sweep_str.replace(c, "")
  return sweep_str


# Not sure of the name: `DirectoriesContext`, `DirBuilder` ?
@dataclasses.dataclass(frozen=True, kw_only=True)
class DirectoryBuilder:
  """Directory factory."""

  subdir_format: SubdirFormat
  unresolved_root_dir: str | None  # At creation time
  ctx: DirContext = dataclasses.field(default_factory=DirContext)

  @functools.cached_property
  def root_dir(self) -> str:
    """Resolve and return the `root_dir`."""
    if self.unresolved_root_dir is None:
      raise ValueError("`xp.root_dir` was not set.")
    return self.format_dir(self.unresolved_root_dir)

  @functools.cached_property
  def xp_dir(self) -> str:
    """Resolve and return the `xp_dir`."""
    if self.ctx.xp is _MISSING:
      raise ValueError("Cannot use `xp_dir` before experiment was created.")
    xp_dir = os.path.join(self.root_dir, self.subdir_format.xp_dirname)
    xp_dir = self.format_dir(xp_dir)
    # TODO(b/312394098): This isn't working on Colab !!
    exm.add_experiment_artifact("Workdir", xp_dir)
    return xp_dir

  @functools.cached_property
  def wu_dir(self) -> str:
    """Resolve and return the `wu_dir`."""
    if self.ctx.wu is _MISSING:
      raise ValueError("Cannot use `wu_dir` before work-unit was created.")
    wu_dir = os.path.join(self.xp_dir, self.subdir_format.wu_dirname)
    wu_dir = self.format_dir(wu_dir)
    exm.add_work_unit_artifact("Workdir", wu_dir)
    return wu_dir

  def replace_ctx(self, **ctx_kwargs) -> DirectoryBuilder:
    """Update the context with new values."""
    return dataclasses.replace(
        self, ctx=dataclasses.replace(self.ctx, **ctx_kwargs)
    )

  def format_dir(self, dir_str: str) -> str:
    return dir_str.format_map(self)

  def __getitem__(self, name: str) -> str:
    if not hasattr(self.subdir_format, name):
      raise ValueError(f"Unrecognized dir property: {{{name}}}.")

    value_fn = getattr(self.subdir_format, name)
    try:
      object.__setattr__(self.ctx, "_ensure_not_missing", True)
      value = value_fn(self.ctx)
    except Exception as e:  # pylint: disable=broad-exception-caught
      epy.reraise(e, f"Error while parsing: {e}")
    finally:
      object.__setattr__(self.ctx, "_ensure_not_missing", False)
    return value


def file_path(name: str) -> str:
  """Returns a proxy string, replaced later by `Fileset().get_path(name)`."""
  return f"__kxm_file__({name})"


# Proxy string, replaced later by the real resolved value.
WU_DIR_PROXY = "__kxm_wu_workdir__"
XP_DIR_PROXY = "__kxm_xp_workdir__"
