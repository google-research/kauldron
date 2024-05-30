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

"""Kauldron-specific utils.

This module contain the Kauldron-specific code.
The rest of the `kxm` implementation is completely independent of Kauldron.

This can serve as example of how `kxm` can be customized for another project.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
import dataclasses
import functools
import itertools
import json
import operator
import types
import typing
from typing import Self

from etils import epath
from etils import exm
from kauldron import konfig
from kauldron.xm._src import cfg_provider_utils
from kauldron.xm._src import dir_utils
from kauldron.xm._src import experiment
from kauldron.xm._src import job_lib
from kauldron.xm._src import job_params
from kauldron.xm._src import jobs_info
from kauldron.xm._src import merge_utils
from kauldron.xm._src import run_strategies
from kauldron.xm._src import sweep_utils
from xmanager import xm

if typing.TYPE_CHECKING:
  from kauldron import kd  # pylint: disable=g-bad-import-order  # pytype: disable=import-error

# Flag to send the json-serialized sweep overwrites kwargs
# If modifying this, also modify the value in `kauldron/utils/sweep_utils.py`
SWEEP_FLAG_NAME = "sweep_config"

# TODO(epot): Support sweep on platform,...


class KauldronJobs(jobs_info.JobsProvider):
  """Extract jobs from a Kauldron `kd.train.Trainer` config."""

  @functools.cached_property
  def jobs(self) -> dict[str, job_lib.Job]:
    return {
        "train": self.trainer_job,
        **self.eval_jobs,
    }

  @functools.cached_property
  def eval_jobs(self) -> dict[str, job_lib.Job]:
    """Returns the evaluation runtime info."""
    # Resolve configs
    runs = {
        eval_name: _resolve_run_konfig(eval_cfg.run)
        for eval_name, eval_cfg in self.cfg_provider.config.evals.items()
        if "run" in eval_cfg
    }

    # Merge shared run
    final_runs = {}
    run_to_eval_names = collections.defaultdict(list)
    run_to_final_eval_names = collections.defaultdict(list)
    for eval_name, run in runs.items():
      if isinstance(run, run_strategies.RunSharedXM):
        # TODO(epot): Validate that the runtimes are the same.
        # Currently, the naive equality fail because `xm.JobRequirement` and
        # `xm_abc.Borg` don't support `__eq__`.
        # if (prev_run := final_runs.get(run.shared_name)) and prev_run != run:
        #   raise ValueError(
        #       "Inconsistent RunSharedXM: the shared runtime from"
        #       f" {eval_name} is different from the ones in"
        #       f" {run_to_eval_names[run.shared_name]}."
        #   )

        final_runs[run.shared_name] = run
        if run.final_eval:
          run_to_final_eval_names[run.shared_name].append(eval_name)
        else:
          run_to_eval_names[run.shared_name].append(eval_name)
      elif isinstance(run, run_strategies.RunXM):
        final_runs[eval_name] = run
        run_to_eval_names[eval_name].append(eval_name)
      elif isinstance(
          run, (run_strategies.RunEvery, run_strategies.RunOnce)
      ):  # Filter run-every
        pass
      else:
        raise TypeError(
            f"Unexpected run strategy for {eval_name}. Got: {type(run)}."
        )

    # Create the associated job
    job = {}
    for eval_name, run in final_runs.items():
      eval_names = run_to_eval_names[eval_name]
      final_eval_names = run_to_final_eval_names[eval_name]
      args = {}
      if eval_names:
        args["eval_names"] = ",".join(eval_names)
      if final_eval_names:
        args["final_eval_names"] = ",".join(final_eval_names)
      job[eval_name] = merge_utils.merge(
          self.base_job,
          run,
          job_params.JobParams(args=args),
      )
    return job

  @functools.cached_property
  def trainer_xm_job(self) -> job_lib.Job:
    return konfig.resolve(self.cfg_provider.config.xm_job)

  @functools.cached_property
  def trainer_job(self) -> job_lib.Job:
    return merge_utils.merge(self.base_job, self.trainer_xm_job)

  @functools.cached_property
  def base_job(self) -> job_lib.Job:
    return job_lib.Job(  # pytype: disable=wrong-keyword-args
        target=self.project_info.target,
        interpreter_info=job_params.InterpreterInfo(
            # We need to explicitly set the script path because the `:trainer`
            # target link to a generated file (from `kauldron_binary`)
            script_path="//third_party/py/kauldron/main.py",
        ),
        args={
            "cfg": cfg_provider_utils.CFG_FLAG_VALUES,
            "cfg.workdir": dir_utils.WU_DIR_PROXY,
        },
    )

  def experiment_creation(self, xp: xm.Experiment) -> None:
    if xp.context.annotations.title == experiment.DEFAULT_EXPERIMENT_NAME:
      xp.context.annotations.set_title(
          f"{self.project_info.project_name}.{self.cfg_provider.config_path.stem}"
      )

    super().experiment_creation(xp)

    if self.project_info.project_name:
      xp.context.annotations.add_tags(self.project_info.project_name)

    module_name = self.cfg_provider.module.__name__

    template_params = {
        "SOURCE": f"xid/{xp.experiment_id}",
        "CONFIG_IMPORT": module_name,
    }
    url = f"http://https://kauldron.rtfd.io/en/latest-test#templateParams={json.dumps(template_params)}"
    exm.add_experiment_artifact("https://kauldron.rtfd.io/en/latest-test (Colab)", url)

  @functools.cached_property
  def project_info(self) -> _ProjectInfo:
    """Project name."""
    # If the target is explicitly defined in the config, use that
    if target := self.trainer_xm_job.target:
      # Extract `//path/to/my_project:trainer` -> `my_project`
      project_name = target.rpartition(":")[0].rpartition("/")[-1]
      return _ProjectInfo(target=target, project_name=project_name)

    path = epath.resource_path(self.cfg_provider.module)

    for curr_dir in _iter_parents(path):
      build_path = curr_dir / "BUILD"
      if not build_path.exists():
        continue
      if "\nkauldron_binary(" not in build_path.read_text():
        continue
      return _ProjectInfo(
          target=f"//{epath.relative_to_g3(curr_dir)}:trainer",
          project_name=curr_dir.name,
      )
    else:
      raise ValueError(
          "Could not auto-infer the project from the config path:"
          f" {self.cfg_provider.config_path}. You might have to explicitly"
          " specify `cfg.xm_job.target =`"
      )


def _resolve_run_konfig(
    run: konfig.ConfigDictLike[run_strategies.RunStrategy],
) -> run_strategies.RunStrategy:
  # TODO(epot): Should add another registration mechanism to automatically
  # rewrite the imports.
  if run.__qualname__.startswith("kauldron.kd:evals."):  # pytype: disable=attribute-error
    _, _, end = run.__qualname__.rpartition(".")  # pytype: disable=attribute-error
    run.__qualname__ = f"kauldron.xm._src.run_strategies:{end}"
  return konfig.resolve(run)


@dataclasses.dataclass(frozen=True, kw_only=True)
class KauldronSweep(sweep_utils.SweepInfo):
  """Kauldron sweep.

  Run the named sweeps defined by `sweep_[NAME]()` in the config file.
  If multiple sweep names are given run all their combinations (product).
  Empty string match `def sweep()` (default).
  """
  # Module from which to extract the sweep functions, automatically set inside
  # `replace_with_job_provider`
  _module: types.ModuleType = dataclasses.field(  # pytype: disable=annotation-type-mismatch
      default=None,
      repr=False,
  )

  def __iter__(self) -> Iterable[sweep_utils.SweepItem]:
    assert self._module is not None
    yield from _sweeps_from_module(
        module=self._module,  # pylint: disable=attribute-error
        names=self.sweep_names,
    )

  @functools.cached_property
  def sweep_names(self) -> list[str]:
    match self._sweep_value:
      case None | False:
        return []
      case True:
        return [""]
      case "*":  # All `sweep_` functions
        return all_available_sweep_names(self._module)
      case str():
        return self._sweep_value.split(",")
      case list():
        return self._sweep_value
      case _:
        raise ValueError(f"Unexpected sweep value: {self._sweep_value}")

  @functools.cached_property
  def tags(self) -> list[str]:
    return [f"🧹{name}" for name in self.sweep_names]

  def replace_with_jobs_provider(
      self, jobs_provider: jobs_info.JobsProvider
  ) -> Self:
    if jobs_provider is None:
      # `jobs_provider` is set later in `launch.py` (as it is defined by the
      # `--cfg` flag). Could design a more robust way to do this.
      return self
    if not isinstance(jobs_provider, KauldronJobs):
      raise TypeError(
          "`KauldronSweep` should be used with `KauldronJobs`. Got:"
          f" {type(jobs_provider)}"
      )
    return dataclasses.replace(self, _module=jobs_provider.cfg_provider.module)


def _sweeps_from_module(
    module: types.ModuleType, names: list[str]
) -> Iterable[sweep_utils.SweepItem]:
  # Step 1: Collect all sweep functions
  sweeps = [_get_sweep_fn(module, name) for name in names]

  # Step 2: Merge all sweep functions with product
  for sweep_kwargs in itertools.product(*sweeps):
    sweep_kwargs = functools.reduce(operator.ior, sweep_kwargs, {})
    yield sweep_utils.SweepItem(
        # Use custom encoder to support ConfigDict objects
        job_kwargs={SWEEP_FLAG_NAME: _JsonEncoder().encode(sweep_kwargs)},
        xm_ui_kwargs={k: _ui_repr(v) for k, v in sweep_kwargs.items()},
    )


def _ui_repr(v):
  """Parameters displayed on the UI."""
  # TODO(epot): In theory, could list exhaustivelly all accepted types
  if isinstance(v, (bool, int, str, float, type(None))):
    return v
  repr_ = repr(v)
  if isinstance(v, konfig.ConfigDict):
    repr_ = repr_.removeprefix("<ConfigDict[").removesuffix("]>")
  # TODO(epot): If str is too big, should truncate ?
  return repr_


def _get_sweep_fn(module: types.ModuleType, fn_name: str):
  fn_name = "sweep_" + fn_name if fn_name else "sweep"
  fn = getattr(module, fn_name, None)
  if fn is None:
    available_sweeps = all_available_sweep_names(module)
    raise ValueError(
        f"Could not find sweep function '{fn_name}()' in {module}."
        f" Available sweeps: {available_sweeps}"
    )
  return fn()


def all_available_sweep_names(module: types.ModuleType) -> list[str]:
  """Returns all available sweep names."""
  sweep_names = []
  for name in dir(module):
    if name.startswith("sweep_"):
      sweep_names.append(name.removeprefix("sweep_"))
    elif name == "sweep":
      sweep_names.append("")
  return sweep_names


@dataclasses.dataclass(frozen=True, kw_only=True)
class _ProjectInfo:
  target: str
  project_name: str


class _JsonEncoder(json.JSONEncoder):

  def default(self, o):
    if isinstance(o, konfig.ConfigDict):
      return json.loads(o.to_json())
    else:
      return super().default(o)


def _iter_parents(path: epath.Path) -> Iterable[epath.Path]:
  yield path
  for path in path.parents:
    yield path


def _is_standalone_eval(
    eval_cfg: konfig.ConfigDictLike[kd.evals.EvaluatorBase],
) -> bool:
  """Infer if the config should be launched as standalone job."""
  # Because Kauldron is not imported during resolving the konfig import,
  # it's not clear how to have a good way to detect whether the eval is
  # standalone or inlined with train.
  if not hasattr(eval_cfg, "run"):
    return False
  return eval_cfg.run.__qualname__ == "kauldron.kd:evals.RunXM"
