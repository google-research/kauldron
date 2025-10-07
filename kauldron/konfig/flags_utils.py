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

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
# pylint: disable=protected-access
# pylint: disable=invalid-name

from absl import flags
from kauldron.konfig import configdict_base
from kauldron.konfig import module_configdict
from ml_collections.config_flags import config_flags


class RecordConfigFileFlagParser(config_flags.ConfigFileFlagParser):
  """A ConfigFileFlagParser that simply records input arguments.

  It creates a konfig.ModuleConfigDict with the input arguments.
  """

  def parse(self, path):
    """Creates a ModuleConfigDict from a config file path."""
    if ":" in path:
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


class LazyConfigFlag(config_flags._ConfigFlag):
  """A ConfigFlag that is lazily resolved.

  The underlying _value field should be a konfig.ModuleConfigDict.
  """

  @property
  def value(self):
    assert isinstance(
        self._value, module_configdict.ModuleConfigDict
    ), "The value of the flag is not a ModuleConfigDict."
    return self._value.read_and_build()

  @value.setter
  def value(self, val):
    self._value = val


def DEFINE_config_file(
    name: str,
    default: str | None = None,
    help_string: str = "path to config file.",
    *,
    flag_values: flags.FlagValues = flags.FLAGS,
    lock_config: bool = False,
    lazy_resolution: bool = False,
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
    lock_config: If set to True, loaded config will be locked through calling
      .lock() method on its instance (if it exists). (default: False)
    lazy_resolution: If set to True, the config module will be lazily resolved.
      By default, only cfg is lazily resolved.

  Returns:
    a handle to defined flag.
  """
  if lazy_resolution or (name == "cfg"):

    parser = RecordConfigFileFlagParser(name=name, lock_config=lock_config)
    serializer = flags.ArgumentSerializer()
    flag = LazyConfigFlag(
        parser=parser,
        serializer=serializer,
        name="cfg",
        default=default,
        help_string=help_string,
        flag_values=flag_values,
        accept_new_attributes=True,
    )

    flag_holder = flags.DEFINE_flag(flag, flag_values)
    return flag_holder
  else:
    return config_flags.DEFINE_config_file(
        name,
        default,
        help_string,
        flag_values=flag_values,
        lock_config=lock_config,
        accept_new_attributes=True,
    )
