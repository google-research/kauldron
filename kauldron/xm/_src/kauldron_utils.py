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
import json
import typing
from typing import Self

from etils import epath
from etils import epy
from etils import exm
from kauldron import konfig
from kauldron import kontext
from kauldron.evals import run_strategies
from kauldron.xm._src import cfg_provider_utils
from kauldron.xm._src import dir_utils
from kauldron.xm._src import experiment
from kauldron.xm._src import job_lib
from kauldron.xm._src import job_params
from kauldron.xm._src import jobs_info
from kauldron.xm._src import merge_utils
from kauldron.xm._src import sweep_cfg_utils
from kauldron.xm._src import sweep_utils
from xmanager import xm

if typing.TYPE_CHECKING:
  from kauldron import kd  # pylint: disable=g-bad-import-order  # pytype: disable=import-error

_Json = epy.typing.Json

# TODO(epot): Support sweep on platform,...


class KauldronJobs(jobs_info.JobsProvider):
  """Extract jobs from a Kauldron `kd.train.Trainer` config."""

  @functools.cached_property
  def _cfg(self) -> kd.train.Trainer:
    """Trainer config."""
    return self.cfg_provider.config

  @functools.cached_property
  def jobs(self) -> dict[str, job_lib.Job]:
    jobs = {}
    if not self._is_eval_only:
      jobs["train"] = self.trainer_job
    jobs.update(self.eval_jobs)
    return jobs

  @functools.cached_property
  def eval_jobs(self) -> dict[str, job_lib.Job]:
    """Returns the evaluation runtime info."""
    # Resolve configs
    runs = {
        eval_name: self._resolve_run_konfig(eval_name, eval_cfg.run)
        for eval_name, eval_cfg in self._cfg.evals.items()
        if "run" in eval_cfg
    }

    # Merge shared run
    final_runs = {}
    run_to_eval_names = collections.defaultdict(list)
    for eval_name, run in runs.items():
      if isinstance(run, run_strategies.Standalone):
        # TODO(epot): Merge all `Standalone` jobs into a single `JobParams`.
        # This would allow to share `StandaloneLastCheckpoint` and
        # `StandaloneEveryCheckpoint` on the same group, as well as partially
        # defining different requirements for each evals. And ensure the
        # params are consistent with each others.

        job_group = run.job_group or eval_name
        final_runs[job_group] = run
        run_to_eval_names[job_group].append(eval_name)
      elif isinstance(run, run_strategies.AlongTrain):  # Filter run-every
        pass
      else:
        raise TypeError(
            f"Unexpected run strategy for {eval_name}. Got: {run!r}."
        )

    # Create the associated job
    return {
        eval_name: merge_utils.merge(
            self.base_job,
            run,
            job_params.JobParams(
                args={"eval_names": ",".join(eval_names)},
            ),
        )
        for eval_name, (run, eval_names) in epy.zip_dict(
            final_runs, run_to_eval_names
        )
    }

  @functools.cached_property
  def trainer_xm_job(self) -> job_lib.Job:
    return konfig.resolve(self._cfg.xm_job)

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
    # Should we add an `eval` tag when job is `eval_only` ? Likely redundant
    # with the config name.

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

  @functools.cached_property
  def _is_eval_only(self) -> bool:
    return kontext.get_by_path(self._cfg, "setup.eval_only", False)

  def _resolve_run_konfig(
      self,
      eval_name: str,
      run: konfig.ConfigDictLike[run_strategies.RunStrategy],
  ) -> run_strategies.RunStrategy:
    # We do not want to trigger a full Kauldron import, so we rewrite the
    # import path.
    # TODO(epot): Should add another registration mechanism to automatically
    # rewrite the imports.
    if run.__qualname__.startswith("kauldron.kd:evals."):  # pytype: disable=attribute-error
      _, _, end = run.__qualname__.rpartition(".")  # pytype: disable=attribute-error
      run.__qualname__ = f"kauldron.evals.run_strategies:{end}"
    run = konfig.resolve(run)
    # Resolve the XM parameters of the `Standalone` jobs. Those had to be
    # defined in `__konfig_resolve_exclude_fields__` to avoid trigger
    # slow XManager import in Kauldron.
    if isinstance(run, run_strategies.Standalone):
      init_kwargs = konfig.resolve(run._kxm_init_kwargs)  # pylint: disable=protected-access
      # Cannot use `dataclasses.replace` as it interact with the `merge_utils`
      run = type(run)(**init_kwargs)

    if self._is_eval_only:  # Eval only, normalize run config.
      if isinstance(run, run_strategies.AlongTrain):
        run = run_strategies.StandaloneLastCheckpoint(
            # `eval_only` likely won't colide with other `cfg.evals` names.
            job_group="eval_only",
            **self.trainer_xm_job._kxm_init_kwargs,  # pylint: disable=protected-access
        )
      elif isinstance(run, run_strategies.Standalone):

        run = run_strategies.StandaloneLastCheckpoint(
            job_group=run.job_group,
            **run._kxm_init_kwargs,  # pylint: disable=protected-access
        )
      else:
        raise TypeError(
            f"Unexpected run strategy for {eval_name}. Got: {type(run)}."
        )
    return run


@dataclasses.dataclass(frozen=True, kw_only=True)
class KauldronSweep(sweep_cfg_utils.SweepFromCfg):
  """Kauldron sweep.

  Run the named sweeps defined by `sweep_[NAME]()` in the config file.
  If multiple sweep names are given run all their combinations (product).
  Empty string match `def sweep()` (default).
  """

  def __iter__(self) -> Iterable[sweep_utils.SweepItem]:
    assert self._module is not None
    for sweep_item in super().__iter__():
      yield _encode_sweep_item(sweep_item)

  def replace_with_jobs_provider(
      self, jobs_provider: jobs_info.JobsProvider
  ) -> Self:
    if not isinstance(jobs_provider, KauldronJobs):
      raise TypeError(
          "`KauldronSweep` should be used with `KauldronJobs`. Got:"
          f" {type(jobs_provider)}"
      )
    return super().replace_with_jobs_provider(jobs_provider)


def _encode_sweep_item(
    sweep_item: sweep_utils.SweepItem,
) -> sweep_utils.SweepItem:
  """Encodes the sweep args."""
  job_kwargs = sweep_item.job_kwargs
  return dataclasses.replace(
      sweep_item,
      # Use custom encoder to support ConfigDict objects
      job_kwargs=_serialize_job_kwargs(job_kwargs),
      xm_ui_kwargs={k: _ui_repr(v) for k, v in job_kwargs.items()},
  )


def _serialize_job_kwargs(job_kwargs: dict[str, _Json]) -> dict[str, _Json]:
  return {
      f"cfg.{k}": v if isinstance(v, str) else _JsonEncoder().encode(v)
      for k, v in job_kwargs.items()
  }


def deserialize_job_kwargs(job_kwargs: dict[str, _Json]) -> dict[str, _Json]:
  return {
      k.removeprefix("cfg."): _decode_json_or_str(v)
      for k, v in job_kwargs.items()
  }


def _decode_json_or_str(v: _Json) -> _Json:
  """Decodes the JSON string or returns the string itself."""
  # The decoded values should always have been encoded JSON strings from
  # `_serialize_job_kwargs`, so there shouldn't be risk of badly formatted JSON.
  try:
    return json.loads(v)
  except json.JSONDecodeError:
    return v


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
