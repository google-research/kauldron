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

"""Jobs with a `--cfg.xxx` flag."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import os
import pathlib
import types
import typing
from typing import Any, Self

from absl import flags
from etils import epath
from kauldron import konfig
from kauldron import kontext
from kauldron.xm._src import dir_utils
from xmanager import xm

if typing.TYPE_CHECKING:
  from kauldron.xm._src import job_lib  # pylint: disable=g-bad-import-order
  import ml_collections  # pylint: disable=g-bad-import-order

# TODO(epot): Support sweep on platform,...

CFG_FLAG_VALUES = "__kxm_cfg_flag_values__"


@dataclasses.dataclass(frozen=True)
class ConfigProviderBase:

  def maybe_add_cfg_flags(self, job: job_lib.Job) -> job_lib.Job:
    raise NotImplementedError()

  def experiment_creation(self, xp: xm.Experiment) -> None:
    pass


@dataclasses.dataclass(frozen=True)
class EmptyConfigProvider(ConfigProviderBase):
  """No configs provided."""

  def maybe_add_cfg_flags(self, job: job_lib.Job) -> job_lib.Job:
    return job


@dataclasses.dataclass(frozen=True)
class ConfigProvider(ConfigProviderBase):
  """ConfigDict to add to the jobs.

  This function will propagate the `--cfg.xxx` flag values from the XM
  `launch.py` and propagate it to all jobs which have
  `args={'cfg': kxm.CFG_FLAG_VALUES}`.

  Attributes:
    config: The `ConfigDict` to launch (before resolve)
    config_parameter: Additional parameters that are appended to the config
      path. config-dict supports providing additional parameters to the config
      like this `config.py:args`.
    overrides: Optional `ConfigDict` overwrides (e.g. `{'batch_size': 64}`)
    module: Module containing the config.
  """

  config: konfig.ConfigDictLike[Any]
  _: dataclasses.KW_ONLY
  config_parameter: str | None = None
  overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
  # TODO(epot): Make `module` optional. Currently required to ship the
  # config to the trainer. But could have a fully-serializable mode.
  module: types.ModuleType

  @classmethod
  def from_module(
      cls,
      module: str | types.ModuleType,
      *,
      overrides: dict[str, Any] | None = None,
      config_parameter: str | None = None,
  ) -> ConfigProviderBase:
    """Create a `ConfigProvider` from a config module."""
    if isinstance(module, str):
      module = importlib.import_module(module)
    elif not isinstance(module, types.ModuleType):
      raise TypeError(f"Expected module. Got: {type(module)}")

    if config_parameter is None:
      config = module.get_config()
    else:
      config = module.get_config(config_parameter)
    return cls(
        module=module,
        config=config,
        config_parameter=config_parameter,
        overrides=overrides or {},
    )

  @classmethod
  def from_flag(
      cls,
      flag: flags.FlagHolder[ml_collections.ConfigDict],
  ) -> Self:
    """Create a `ConfigProvider` from a `DEFINE_config_file` flag."""
    # Get the flags.FLAGS object linked to the flag
    flagvalues = flag._flagvalues  # pylint: disable=protected-access

    # Getting the path is tricky because we use DEFINE_config_file for the flag
    # And this flag returns the evaluated config directly instead of the
    # filepath.
    # We cannot simply use a DEFINE_string flag instead, because we also need
    # the evaluated config with CLI config overrides. Thus the hack below:
    config_path = flagvalues[flag.name].config_filename  # pytype: disable=attribute-error
    # DEFINE_config_file supports additional arguments to be appended like this
    # config.py:args
    # We need to remove them to get the actual config path.
    config_path, *config_parameter = config_path.split(":", 1)

    # Import the config module (needed for sweeps etc.).
    config_path = config_path.replace("/", ".")
    config_module = importlib.import_module(config_path)

    # In addition to the filename we also need the config overrides to pass on
    # to the worker units, so here we collect them from the list of all flags.
    config_overrides = {
        flag_name.removeprefix(f"{flag.name}."): flagvalues[flag_name].value
        for flag_name in flagvalues
        if flag_name.startswith(f"{flag.name}.")
    }
    return cls(
        module=config_module,
        config=flag.value,
        config_parameter=config_parameter[0] if config_parameter else None,
        overrides=config_overrides,
    )

  def __post_init__(self) -> None:
    # TODO(epot): I don't think this should be a limitation if `v == 'None'` str
    for k, v in self.overrides.items():
      if v is None:
        raise ValueError(
            f"Value is `None` for parameter {k}. XManager does not support "
            "overriding parameters to `None` and will silently keep the "
            "default value. Note that, in some places, even xm2a/ will be "
            "misleading about this. If you need this and think it should "
            "work, please reach out."
        )

    # Apply the `overrides` as they can contain info on XM `--cfg.xm_job....`
    for k, v in self.overrides.items():
      kontext.set_by_path(self.config, k, v)

  @functools.cached_property
  def config_path(self) -> pathlib.Path:
    """Config path."""
    config_path = pathlib.Path(self.module.__file__)
    return config_path

  def maybe_add_cfg_flags(self, job: job_lib.Job) -> job_lib.Job:
    """Maybe replace `CFG_FLAG_VALUES` by the actual config flags."""
    flag_arg = [k for k, v in job.args.items() if v == CFG_FLAG_VALUES]
    if not flag_arg:  # No `CFG_FLAG_VALUES`
      return job
    elif len(flag_arg) > 1:
      raise ValueError(
          f"Multiple `CFG_FLAG_VALUES` found. This is not supported. {flag_arg}"
      )
    (flag_arg,) = flag_arg

    # Is the file dependency really needed? Likely not as it would not
    # work for configs depending on multiple files.
    _CFG_FILENAME = "config.py"  # pylint: disable=invalid-name

    cfg_path = dir_utils.file_path(_CFG_FILENAME)
    if self.config_parameter is not None:
      cfg_path += f":{self.config_parameter}"

    new_args = dict(job.args)
    new_args[flag_arg] = cfg_path
    # TODO(epot): Should ensure no collision with existing flags.
    new_args.update({f"{flag_arg}.{k}": v for k, v in self.overrides.items()})

    return job.replace(
        args=new_args,
        files=job.files | {_CFG_FILENAME: f"//{self.config_path}"},
    )

  def experiment_creation(self, xp: xm.Experiment) -> None:
    from kauldron.xm._src import experiment  # pylint: disable=g-import-not-at-top

    if xp.context.annotations.title == experiment.DEFAULT_EXPERIMENT_NAME:
      xp.context.annotations.set_title(self.config_path.stem)

    xp.context.add_config_file(
        file_content=inspect.getsource(self.module),
        description=f"Content of {self.config_path}",
    )
