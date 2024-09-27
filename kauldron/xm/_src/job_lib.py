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

import asyncio
import dataclasses
import functools
import shlex
import typing
from typing import Any, Optional

import attr
from etils import etree
from etils import g3_utils
from kauldron.xm._src import dir_utils
from kauldron.xm._src import job_params
from xmanager import resource_selector as rs
from xmanager import xm
from xmanager import xm_abc  # Open-source would use xm_local
from xmanager.contrib.internal import requirements_flag as xmreq
from xmanager.contrib.internal import xm_jax


# repr=False to use the parent `__repr__`
@dataclasses.dataclass(frozen=True, kw_only=True, repr=False)
class Job(job_params.JobParams):
  """Single XManager job.

  All customizable attributes are defined and documented in the parent
  `JobParams` class.
  """

  # Job name, automatically set
  name: None | str = None

  # Automatically set after `queue_build()` is called
  executable_future: asyncio.Future[xm.Executable] | None = dataclasses.field(
      default=None, repr=False
  )

  # Do NOT add new attributes here. Job attributes are defined in the base
  # class `JobParams`

  @property
  def executable(self) -> xm.Executable:
    """Returns the executable."""
    if self.executable_future is None:
      raise ValueError(
          "`job.executable` should only be called after `queue_build()`"
      )
    # XM return a `PicklableAwaitableImpl` so have to unwrap it
    return self.executable_future._get_future().result()  # pylint: disable=protected-access  # pytype: disable=attribute-error

  def queue_build(self) -> None:
    """Add the job in the build queue."""
    if not self.target:
      raise ValueError(f"Missing `target=` for job {self.name!r}.")

    # TODO(epot): Support MPM
    if self.use_interpreter:
      package = self.interpreter_package
    else:
      package = self.bazel_package

    executable_future = xm_abc.get_current_experiment().package_async(package)
    object.__setattr__(self, "executable_future", executable_future)

  @functools.cached_property
  def bazel_package(self) -> xm.Packageable:
    """Returns the bazel package."""
    requirements = typing.cast(xm.JobRequirements, self.requirements)
    bazel_args = xm_abc.bazel_args.for_resource(requirements.accelerator)
    # This is necessary for linking to succeed when using GPUs.
    # In some informal testing, it did not seem to slow down compilation of
    # non-GPU workloads, so we're always enabling it for simplicity.
    bazel_args = bazel_args + ("--define=cuda_compress=1",)

    # Add any additional bazel args specified in JobParams.
    bazel_args = bazel_args + tuple(self.bazel_args)

    # Not sure where the original =TRUE comes from, but it needs overwriting.
    if not self.debug.pytype:
      bazel_args = tuple(
          "--define=PYTYPE=FALSE" if a == "--define=PYTYPE=TRUE" else a
          for a in bazel_args
      )
    return xm.bazel_binary(
        label=self.target,
        dependencies=self.dependencies,
        executor_spec=xm_abc.Borg.Spec(),
        # Need to wait for next build
        bazel_args=bazel_args,
    )

  @functools.cached_property
  def interpreter_package(self) -> xm.Packageable:
    """Returns the interpreter package."""
    if self.interpreter_info.script_path:
      script_path = self.interpreter_info.script_path
    else:
      script_path = _target_to_script_path(self.target)

    # TODO(epot): `merge_utils` should support `xm_abc.ml_python`, then
    # here, should replace
    # `xm_abc.ml_python().replace(accelerator=self.requirements.accelerator)`

    # Set the adhoc import dir to the user workspace (or adhoc import source
    # in Colab)
    citc_info = g3_utils.citc_info_from_source_or_piper(self.citc_source)
    citc_info = citc_info.immutable()

    script_full_path = citc_info.g3_path / script_path.removeprefix("//")
    if not script_full_path.exists():
      raise ValueError(f"Could not find script {script_full_path}.")

    mpm = self.interpreter_info.mpm
    if isinstance(mpm, job_params.MLPython):
      mpm = mpm.get_mpm(accelerator=self.requirements.accelerator)

    return xm_abc.interpreter(  # pytype: disable=wrong-arg-types
        script_path=script_path,
        interpreter_mpm=mpm,
        dependencies=self.dependencies,
        args={
            "adhoc_import_dir": citc_info.g3_dir,
        },
    )

  @functools.cached_property
  def rs_job(self) -> rs.Job:
    """Resource requirements with constraints."""
    constraints = list(self.constraints)
    constraints.append(rs.Borg())  # Currently, only Borg supported
    reqs_kwargs = {"priority": self.priority}
    if self.cell is not None:
      reqs_kwargs["location"] = self.cell
    if self.platform is not None:
      reqs_kwargs.update(xmreq.parse_requirements_spec(self.platform))
    # TODO(epot): Should have `xm.JobRequirements` everywhere, rather than
    # my custom implementation. Send a cl to `xm` to have default args being
    # missing ?
    job_requirements = xm.JobRequirements(
        **reqs_kwargs,
        **self.requirements._kxm_init_kwargs,  # pylint: disable=protected-access  # pytype: disable=attribute-error
    )
    return rs.Job(
        constraints=constraints,
        requirements=job_requirements,
    )

  def make_xm_job(
      self,
      *,
      sweep_args: dict[str, Any],
      dir_builder: dir_utils.DirectoryBuilder,
  ) -> xm.Job:
    """Returns the XManager job."""

    # Build up the environment variables:
    env_vars = dict(self.env_vars)
    if self.debug.dump_hlo:
      xla_flags = env_vars.get("XLA_FLAGS", "")
      xla_flags += f" --xla_dump_to={dir_builder.wu_dir}"
      env_vars["XLA_FLAGS"] = xla_flags
    if self.debug.flax_profile:
      env_vars["FLAX_PROFILE"] = "true"  # Could also be experiment-wide.

    # Build up the args:
    args = dict(self.args)
    args.update(sweep_args)
    args = {
        k: _resolve_and_normalize_arg(
            v, dir_builder=dir_builder, fileset=self.fileset
        )
        for k, v in args.items()
    }

    # Add common args for all jobs.
    if self.debug.catch_post_mortem:
      args["catch_post_mortem"] = True

    # Add optional jax and debug flags.
    if self.add_jax_flags:
      args.update(xm_jax.JaxFlags().flags())
      args["jax_log_compiles"] = True

    if self.debug.xprof_port:
      args["xprof_port"] = "%port_xprof%"
    if self.debug.g3pdb_port:
      args["g3pdb_port"] = "%port_g3pdb%"

    # TODO(b/322769542): Remove once XM fix this issue.
    use_auto_host_resources = _get_use_auto_host_resources(
        executor=self.executor,
        requirements=self.requirements,
    )

    executor = attr.evolve(
        self.executor,
        use_auto_host_resources=use_auto_host_resources,
        requirements=self.requirements,
    )
    return xm.Job(
        executable=self.executable,
        executor=executor,
        args=args,
        env_vars=env_vars,
    )

  @functools.cached_property
  def fileset(self) -> xm_abc.Fileset:
    """Fileset."""
    # Invert key<>value
    return xm_abc.Fileset(files={v: k for k, v in self.files.items()})

  @functools.cached_property
  def dependencies(self) -> list[xm.BinaryDependency]:
    if self.fileset.files:
      return [self.fileset]
    else:
      return []


def _resolve_and_normalize_arg(
    arg: str,
    dir_builder: dir_utils.DirectoryBuilder,
    fileset: xm_abc.Fileset,
) -> str:
  """Build up the commandline arguments to be passed."""
  # Use lambda to lazy-resolve the `dir_builder.wu_dir`, otherwise it would
  # raise an error when `xp.root_dir` is not set even it it's never used.
  replaces = {
      "%": lambda: "%%",  # Fixes an issue with tfds splits containing "%"
      dir_utils.WU_DIR_PROXY: lambda: dir_builder.wu_dir,
      dir_utils.XP_DIR_PROXY: lambda: dir_builder.xp_dir,
  }
  replaces_startswith = {
      dir_utils.file_path(file): fileset.get_path(file, xm_abc.Borg.Spec())
      for file in fileset.files.values()
  }

  def sanitize_leaf(leaf):
    if isinstance(leaf, str):
      for before, after in replaces.items():
        if before in leaf:
          leaf = leaf.replace(before, after())
      # Some file paths can contain additional arguments, e.g. config-dict
      # files support passing arguments separated by a colon:
      # `__xm_file__(config.py):arg`
      # In this case we need to replace the `__xm_file__` prefix with the
      # actual file path, and then add the arguments back.
      for before, after in replaces_startswith.items():
        if leaf.startswith(before):
          suffix = leaf.removeprefix(before)
          if isinstance(after, str):
            leaf = after + suffix
          elif isinstance(after, xm.ShellSafeArg):
            leaf = xm.ShellSafeArg(after.arg + shlex.quote(suffix))
          else:
            raise ValueError(
                f"Unsupported type for `replaces_startswith`: {type(after)}"
            )

    return leaf

  arg = etree.map(sanitize_leaf, arg)

  # Wrap tuples in string so that they are treated as tuples and not
  # repeated values for the same flag.
  if isinstance(arg, tuple):
    arg = str(arg)

  return arg


def _get_use_auto_host_resources(
    executor: xm.Executor,
    requirements: xm.JobRequirements,
) -> Optional[bool]:
  """Returns if `use_auto_host_resources` should be used."""
  if not isinstance(executor, xm_abc.Borg):
    return None  # `use_auto_host_resources` only exists in Borg
  # use_auto_host_resources explicitly set
  if executor.use_auto_host_resources is not None:
    return executor.use_auto_host_resources
  if requirements.accelerator not in xm.GpuType:
    return None  # When not using GPUs, let XM decide
  return not {
      xm.ResourceType.CPU,
      xm.ResourceType.MEMORY,
      xm.ResourceType.RAM,
  } & set(requirements.task_requirements.keys())


def _target_to_script_path(target: str) -> str:
  return target.replace(":", "/").removesuffix(".py") + ".py"
