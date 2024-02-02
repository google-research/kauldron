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

"""XManager Experiment."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import dataclasses
import functools
import os
import typing
from typing import Optional

from etils import epy
from etils import exm
from etils import g3_utils
from kauldron.xm._src import dir_utils
from kauldron.xm._src import job_lib
from kauldron.xm._src import job_params
from kauldron.xm._src import jobs_info as jobs_info_lib
from kauldron.xm._src import merge_utils
from kauldron.xm._src import orchestrator as orchestrator_lib
from kauldron.xm._src import requirements as requirements_lib
from kauldron.xm._src import sweep_utils
from kauldron.xm._src import utils
from xmanager import resource_selector as rs
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib.internal import tensorboard

DEFAULT_EXPERIMENT_NAME = "Kauldron"


@dataclasses.dataclass(frozen=True, kw_only=True)
class Experiment(job_params.JobParams):
  """XManager experiment to launch.

  See base class `JobParams` to set job attributes (cell, platform,...).
  `JobParams` attributes defined here will globally update all
  jobs that do not explicitly set this attribute.

  Attributes:
    name: Name of experiment. If unspecified, use config filename.
    tags: Comma-separated tags, e.g. tag1,tag2
    note: Optional free-form note for the experiment.
    jobs: Direct way to define which jobs to launch
    jobs_provider: Alternative way to provide jobs (e.g. dynamically loaded from
      a config file)
    root_dir: Root directory where to store checkpoints,... Each time an
      experiment is launched, a new `{xid}/` subfolder is created. The
      `root_dir` can contain `{cell}`, `{author}` placeholders.
    subdir_format: Specify the dirname for the xp and work-unit subfolder.
      Default implementation create `{root_dir}/{xid}/{wid}/`
    sweep: Whether or not running the sweep. The value here is propagated to the
      `sweep_info._sweep_value`.
    sweep_info: Sweep kwargs builder. Specify how to get the sweep kwargs.
    orchestrator: Indicate how to launch the jobs (which order,...)
    importance: Experiment importance.
    execution_settings: `xm_abc.ExecutionSettings` of the experiment.
    emoji_tags: Whether to use cool emoji tags.
    add_tensorboard_borg: Add TensorBoard
    add_tensorboard_corp: Add TensorBoard corp
  """

  # Base experiment information
  name: Optional[str] = None
  tags: list[str] = dataclasses.field(default_factory=list)
  note: str = ""

  # Job, sweep & execution
  jobs: dict[str, job_lib.Job] = dataclasses.field(default_factory=dict)
  jobs_provider: jobs_info_lib.JobsProvider = dataclasses.field(
      default_factory=jobs_info_lib.EmptyJobs
  )

  root_dir: str | None = None
  subdir_format: dir_utils.SubdirFormat = dataclasses.field(
      default_factory=dir_utils.SubdirFormat
  )

  sweep: Optional[bool | str | list[str]] = None
  sweep_info: sweep_utils.SweepInfo = sweep_utils.NoSweep()

  orchestrator: orchestrator_lib.Orchestrator = dataclasses.field(
      default_factory=orchestrator_lib.SweepOrchestrator
  )

  # Additional experiment-level info
  importance: xm.Importance = xm.Importance.NORMAL
  execution_settings: xm_abc.ExecutionSettings = dataclasses.field(
      default_factory=xm_abc.ExecutionSettings
  )
  emoji_tags: bool = False

  # Auxiliary units (TB,...)
  add_tensorboard_borg: bool = False
  add_tensorboard_corp: bool = False

  def __post_init__(self):
    super().__post_init__()
    # By default, set read permission to everyone
    if self.execution_settings.read_users is None:
      self.execution_settings.read_users = ["mdb/all"]

    # Support `--xp.tags=my_tag,another_tag`
    if isinstance(self.tags, str):
      object.__setattr__(self, "tags", self.tags.split(","))

    if self.sweep in (None, False):
      new_sweep = sweep_utils.NoSweep()
    else:
      new_sweep = dataclasses.replace(
          self.sweep_info,
          # Some sweep require info from the job builder (e.g. to load from
          #  config). So link the sweep to the job
          _jobs_provider=self.jobs_provider,
          # Propagate the `--xp.sweep=` value.
          _sweep_value=self.sweep,
      )
    object.__setattr__(self, "sweep_info", new_sweep)

  def launch(self) -> xm_abc.XManagerExperiment:
    """Launch the experiment."""
    with self.create_experiment() as xp:
      xp.context.add_config_file(
          file_content=epy.pretty_repr(self.resolved_jobs),
          description="Jobs",
      )

      # Step 1: Collect and build all jobs
      for job in self.resolved_jobs.values():
        job.queue_build()  # Queue jobs with `xp.package_async()`
      with utils.maybe_log_colab():
        xp.package()  # Trigger the `bazel build`

      # Step 2: Launch the jobs (sweep, pipeline,...)
      dir_builder = self.dir_builder.replace_ctx(
          xp=xp,  # Can be used to build the directory (`{cell}`, ...)
          resolved_jobs=self.resolved_jobs,
          sweep_info=self.sweep_info,
      )
      self.orchestrator.launch_jobs(
          resolved_jobs=self.resolved_jobs,
          sweep_info=self.sweep_info,
          dir_builder=dir_builder,
      )

      # Step 3: Launch auxiliaries
      if self.add_tensorboard_borg:
        requirements = xm.JobRequirements(CPU=1)
        tensorboard.add_tensorboard_borg(
            xp,
            workdir=dir_builder.xp_dir,
            executor=xm_abc.Borg(requirements=requirements),
        )
      if self.add_tensorboard_corp:
        requirements = xm.JobRequirements(CPU=1)
        tensorboard.add_tensorboard_corp(
            xp,
            workdir=dir_builder.xp_dir,
            executor=xm_abc.Borg(requirements=requirements),
        )
      # TODO(epot): Support Custom auxiliaries

    return xp

  @contextlib.contextmanager
  def create_experiment(self) -> Iterator[xm_abc.XManagerExperiment]:
    """Wrapper around `xm_abc.create_experiment`."""
    # TODO(epot): Have some heuristic to auto-set the Experiment name from the
    # target ?
    name = self.name or DEFAULT_EXPERIMENT_NAME

    # On Colab, set the g3 dir to the CitC source
    if epy.is_notebook():
      citc_info = g3_utils.citc_info_from_source_or_piper(self.citc_source)
      citc_info = citc_info.immutable()
      g3_dir = citc_info.g3_dir.replace("/google/src", "/google_src", 1)

      os.environ["BUILD_WORKSPACE_DIRECTORY"] = g3_dir

    # Set experiment-level options
    with xm_abc.create_experiment(
        experiment_title=name,
        settings=self.execution_settings,
    ) as xp:
      xp = typing.cast(xm_abc.XManagerExperiment, xp)

      xp.context.annotations.add_tags(*self.all_tags)
      xp.context.annotations.set_notes(self.note)
      xp.context.add_config_file(
          file_content=repr(self),
          description="kxm.Experiment",
      )
      xp.set_importance(self.importance)
      # Let the jobs builder perform additional customization
      self.jobs_provider.experiment_creation(xp)
      yield xp

  @functools.cached_property
  def resolved_jobs(self) -> dict[str, job_lib.Job]:
    """Create all jobs info.

    This function:

    * Resolve all `Job` missing attributes by using the values from the
      `Experiment` level (or default ones).
    * Resolve all constraints (e.g. cell selection).

    Returns:
      jobs: The resolved jobs
    """
    # Provided jobs
    jobs = self.jobs_provider.jobs
    # Set the default name
    jobs = {
        k: j.replace(name=k if j.name is None else j.name)
        for k, j in jobs.items()
    }

    # Merge jobs and jobs_provider
    for k, v in self.jobs.items():
      if k not in jobs:
        jobs[k] = v
      else:
        jobs[k] = merge_utils.merge(jobs[k], v)

    # Merge jobs with the default runtime options
    # We remove the `--xp.name=` kwarg as it is a Experiment level kwarg and
    # create conflicts with `job.name`
    default_job = self.replace(name=dataclasses.MISSING)
    jobs = {
        k: utils.reraise_fn("Error for job {!r}: ", k)(merge_utils.merge)(
            default_job, j
        )
        for k, j in jobs.items()
    }
    # Once the jobs have their properties set, we can resolve all constraints
    # Maybe should try to delay this until the very last moment (to help
    # scheduling match the forecast), but the `cell` resolved here is
    # accessed in many different places.
    requirements = rs.select(*(j.rs_job for j in jobs.values()))
    jobs = {
        k: j.replace(
            # Resolve the `xm.JobRequirements`. From this point, the job
            # info are complete.
            requirements=r,
            # Overwrite the fields that can be set by auto-select.
            cell=r.location,
            platform=requirements_lib.platform_from_requirements(r),
        )
        for (k, j), r in zip(jobs.items(), requirements)
    }

    return jobs

  @functools.cached_property
  def dir_builder(self) -> dir_utils.DirectoryBuilder:
    """Directory state."""
    return dir_utils.DirectoryBuilder(
        unresolved_root_dir=self.root_dir,
        subdir_format=self.subdir_format,
    )

  @functools.cached_property
  def all_tags(self) -> list[str]:
    """Experiment tags."""
    tags = list(self.tags or [])
    tags.append("🍲")  # kauldron!
    if self.emoji_tags:
      make_tag_fn = _make_emoji_tags
    else:
      make_tag_fn = _make_default_tags
    # TODO(epot): Support multi-jobs tag (with name: `train: cpu`,...)
    main_job = list(self.resolved_jobs.values())[0]
    tags.extend(
        make_tag_fn(
            priority=main_job.priority,
            platform=main_job.platform,
            cell=main_job.cell,
        )
    )
    tags.extend(self.sweep_info.tags)
    return tags

  def _repr_html_(self) -> str:
    from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    return ecolab.highlight_html(repr(self))


def _make_default_tags(
    priority: int,
    platform: str,
    cell: Optional[str],
) -> list[str]:
  """Returns the list of default tags."""
  tags = []
  if priority != 200:  # Do not add the default priority
    tags.append(f"p:{priority}")
  tags.append(platform)
  if cell is not None:
    tags.append(f"cell:{cell}")
  return tags


def _make_emoji_tags(
    priority: int,
    platform: str,
    cell: Optional[str],
) -> list[str]:
  """Return a list of fancy emoji tags for priority, platform and cell."""
  tags = []
  if priority < 100:
    tags.append(f"🐶{priority}")
  elif priority < 200:
    tags.append(f"❕{priority}")
  elif priority > 200:
    tags.append(f"❗{priority}")

  if "=" in platform:
    tpu, _, size = platform.partition("=")
    symbol = {
        "jd": "🪼",
        "jf": "🪼",
        "df": "🐲",
        "dd": "🐲",
        "pd": "🐡",
        "pf": "🐡",
        "vl": "🐍",
        "vlp": "🐍",
        "vf": "🐍",
    }.get(tpu, tpu + "=")
    tags.append(f"{symbol}{size}")
  else:
    tags.append(platform)

  if cell is not None:
    tags.append(f"🌎{cell}")
  return tags


merge_utils.add_merge_support(xm.JobRequirements)
merge_utils.add_merge_support(xm_abc.AutopilotParams)
merge_utils.add_merge_support(xm_abc.Borg)
merge_utils.add_merge_support(xm_abc.BorgletParams)
merge_utils.add_merge_support(xm_abc.BorgScheduling)
merge_utils.add_merge_support(xm_abc.ml_python)
