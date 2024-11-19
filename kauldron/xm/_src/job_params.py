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

"""Runtime library."""

from __future__ import annotations

import dataclasses
from typing import Any

from etils import edc
from etils import epy
from kauldron.xm._src import merge_utils
from typing_extensions import Self

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  from xmanager import resource_selector as rs
  from xmanager import xm
  from xmanager import xm_abc  # Open-source would use xm_local
  # pylint: enable=g-import-not-at-top  # pytype: enable=import-error


@merge_utils.add_merge_support
@edc.dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class Debug:
  """Debug options.

  Attributes:
    dump_hlo: Dump all compilation HLO into workdir for debugging.
    flax_profile: Enable annotating every Flax module with a useful string in
      XProf. There are no runtime performance costs but JAX tracing will be ~10%
      slower, and Flax is more strict about function purity.
    pytype: Whether to run pytype (slower, but safer compile).
    catch_post_mortem: Wrap experiment in g3pdb.catch_post_mortem(). In case of
      an uncaught exception this will prevent the experiment from stopping and
      instead start an interactive debugger. Link will be sent by email and
      found among the artifacts.
    xprof_port: Configures xProf port.
    g3pdb_port: Configures g3pdb port.
  """

  dump_hlo: bool = False
  flax_profile: bool = True
  # Pytype is known to be slow (as of 2022Q1), so we disable it by default.
  # A particularly bad example: 290s vs 30s build-time on one-line change.
  pytype: bool = False
  catch_post_mortem: bool = False
  xprof_port: bool = True
  g3pdb_port: bool = True


@merge_utils.add_merge_support
@dataclasses.dataclass(frozen=True, kw_only=True)
class MLPython:
  """Wrapper around `xm_abc.ml_python` that auto-set the accelerator."""

  version: None | str = 'live'  # Set live version because of b/325917292
  fallback_behavior: str = 'live'

  def get_mpm(
      self, accelerator: None | xm.ResourceType = None
  ) -> xm.ExecutableSpec:
    return xm_abc.ml_python(
        version=self.version,
        fallback_behavior=self.fallback_behavior,
        accelerator=accelerator,
    )


@merge_utils.add_merge_support
@dataclasses.dataclass(frozen=True, kw_only=True)
class InterpreterInfo:
  """Interpreter additional configuration.

  Attributes:
    mpm: Intepreter to use when `use_interpreter is True` (default to
      `ml_python`)
    script_path: Path of the script to execute. If missing, is automatically
      computed from the `job.target`.
  """

  mpm: xm.ExecutableSpec | MLPython = dataclasses.field(
      default_factory=MLPython
  )
  script_path: str | None = None


@edc.dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class JobParams:
  """Single job atributes.

  Can be set either:

  * At the `kxm.Experiment` level (applied to all jobs)
  * At the individual `kd.Job` level (explicitly provided in `xp.jobs`)
  * If a `xp.jobs_provider` is provided, their `xp.jobs_provider.jobs` are
    merged with `xp.jobs`

  All attributes are optional. Attribute values are merged across all different
  sources (`xp.jobs`, `xp.jobs_provider`,...). If an attribute is defined twice,
  an error is raised.

  Attributes:
    target: Build target
    platform: Accelerator specification. Eg.: `cpu`, `v100`, `df=4x4`
    priority: Use 200 for PROD, 115 for BATCH and 25 for FREEBIE
    cell: Which cell to run on. If `None`, use auto-select.
    citc_source: On Colab, allow to specify from which citc workspace source
      should the experiment be built, or which adhoc-import path to use for the
      interpreter. Values match `ecolab.adhoc` source.
    use_interpreter: If True, use `ml_python` or the interpreter specified in
      `interpreter_mpm`.
    interpreter_info: Additional interpreter configuration options.
    constraints: Additional scheduling constraints (e.g. cell restrictions)
    requirements: Additional resources requirements (e.g. local disk, ram,...)
    executor: Executor options.
    args: Job arguments
    env_vars: Environement variables of the job
    files: Additional files to include (Mapping from filename to
      pkgdef-reference)
    add_jax_flags: Can be disabled for non-jax jobs.
    bazel_args: Additional job-specific bazel arguments as list of strings.
    debug: Additional debug options.
  """

  # TODO(epot): Support ml_python, MPM,...
  target: str = ''

  priority: int = 200
  platform: None | str = None
  cell: None | str = None

  citc_source: None | g3_utils.Source = None
  use_interpreter: bool = False
  interpreter_info: InterpreterInfo = dataclasses.field(
      default_factory=InterpreterInfo
  )
  constraints: list[rs.Constraint] = dataclasses.field(default_factory=list)
  # Use lambda to not resolve the lazy-imports
  requirements: xm.JobRequirements = dataclasses.field(
      default_factory=lambda: xm.JobRequirements()  # pylint: disable=unnecessary-lambda
  )
  executor: xm.Executor = dataclasses.field(
      default_factory=lambda: xm_abc.Borg()  # pylint: disable=unnecessary-lambda
  )

  args: dict[str, Any] = dataclasses.field(default_factory=dict)
  env_vars: dict[str, str] = dataclasses.field(default_factory=dict)
  # TODO(epot): File API v2:
  # Should support direct file usage, like:
  # `args={'gin_config': kxm.file_path('//path/to/my_file')}`
  # without having to use `files=` attribute first.
  files: dict[str, str] = dataclasses.field(default_factory=dict)

  add_jax_flags: bool = True
  bazel_args: list[str] = dataclasses.field(default_factory=list)
  debug: Debug = dataclasses.field(default_factory=Debug)

  # Internal variable to prevent `dataclasses.replace`
  _replace_sentinel: bool = dataclasses.field(default=False, repr=False)

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    # Set `__repr__` so the `dataclass` don't generate one, it's
    # important the custom repr is used to support pretty display
    # from `merge_utils`
    cls.__repr__ = JobParams.__repr__

  def __post_init__(self):
    # TODO(epot): Should make sure `super().__post_init__` is called in subclass
    if self._replace_sentinel:
      raise ValueError(
          'Calling `dataclasses.replace()` on `JobParams` subclasses is'
          ' forbidden as this interact with the mergin system. Instead,'
          ' `job.replace()` method should be used.'
      )
    object.__setattr__(self, '_replace_sentinel', True)

  def replace(self, **kwargs) -> Self:
    final_kwargs = self._kxm_init_kwargs  # pytype: disable=attribute-error
    final_kwargs.update(kwargs)
    final_kwargs = {  # Filter `dataclasses.MISSING` attributes
        k: v for k, v in final_kwargs.items() if v is not dataclasses.MISSING
    }
    return type(self)(**final_kwargs)

  def __repr__(self) -> str:
    with merge_utils.repr_only_init():
      # Only the second level should be displayed simplified, but not
      # the top level
      return epy.pretty_repr_top_level(self, force=True)


# TODO(epot): Pytype bug and using this as decorator make
# `StandaloneEveryCheckpoint` fail.
merge_utils.add_merge_support(JobParams)
