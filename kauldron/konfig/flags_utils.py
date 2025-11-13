# Copyright 2025 The kauldron Authors.
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

"""Flags utils."""

# pylint: disable=g-multiple-import
# pylint: disable=protected-access

import warnings
from absl import flags
from etils import epy
from kauldron.konfig import configdict_base
from kauldron.konfig import module_configdict
from ml_collections.config_flags import config_flags


def DEFINE_config_file(  # pylint: disable=invalid-name
    name: str,
    default: str | None = None,
    help_string: str = "path to config file.",
    *,
    flag_values: flags.FlagValues = flags.FLAGS,
    lock_config: bool = False,
) -> flags.FlagHolder[configdict_base.ConfigDict]:
  """Defines flag for `ConfigDict`.

  This is similar to `ml_collections.config_flags.DEFINE_config_file`, but
  allows for lazy resolution of the config module. The config is resolved from
  flag values when the `value` property of the defined flag is accessed.

  Args:
    name: Flag name, optionally including extra config after a colon.
    default: Default value of the flag (default: None).
    help_string: Help string to display when --helpfull is called. (default:
      "path to config file.")
    flag_values: FlagValues instance used for parsing. (default:
      absl.flags.FLAGS)
    lock_config: Unused argument that we keep for compatibility.

  Returns:
    a handle to defined flag.
  """
  assert not lock_config, "lock_config is not supported."
  flag = _LazyConfigFlag(
      parser=_RecordConfigFileFlagParser(name=name),
      serializer=flags.ArgumentSerializer(),
      name=name,
      default=default,
      help_string=help_string,
      flag_values=flag_values,
      accept_new_attributes=True,
  )

  flag_holder = flags.DEFINE_flag(flag, flag_values)
  return flag_holder


class _RecordConfigFileFlagParser(flags.ArgumentParser):
  """A ConfigFileFlagParser that simply records input arguments.

  It creates a konfig.ModuleConfigDict with the input arguments.
  """

  def __init__(self, name):
    self.name = name

  def parse(self, path):
    """Creates a ModuleConfigDict from a config file path."""
    if ":" in path:
      # TODO(geco): add a doc and raise warning that this is the old behavior.
      module_path, cli_str_arg = path.split(":", 1)
    else:
      module_path, cli_str_arg = path, ""

    config_module = config_flags._LoadConfigModule(  # pylint: disable=protected-access
        "{}_config".format(self.name), module_path
    )
    modulecfg = module_configdict.ModuleConfigDict(
        module=config_module, cli_str_arg=cli_str_arg
    )
    return modulecfg

  def flag_type(self):
    return "config object"


class _LazyConfigFlag(config_flags._ConfigFlag):
  """A ConfigFlag that is lazily resolved.

  The underlying _value field should be a konfig.ModuleConfigDict.
  """

  @property
  def value(self):
    # because of some absl behavior that copies .value into ._value for testing,
    # we need to be flexible here and just trigger a warning.
    _value = self._value  # pylint: disable=invalid-name
    assert _value is not None, "None for _value is not supported."
    if isinstance(self._value, module_configdict.ModuleConfigDict):
      return _value.module_config
    elif epy.is_test():
      warnings.warn(
          "The value of the flag is not a ModuleConfigDict. This is normal if"
          " happening inside an absl test."
      )
      return _value
    else:
      raise ValueError(
          "The value of the flag should be a ModuleConfigDict. Got"
          f" {self._value}"
      )

  @value.setter
  def value(self, val):
    self._value = val
