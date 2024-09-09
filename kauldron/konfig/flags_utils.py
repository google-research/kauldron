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

"""Flags utils."""

from absl import flags
from kauldron.konfig import configdict_base
from ml_collections import config_flags


def DEFINE_config_file(  # pylint: disable=g-bad-name
    name: str,
    default: str | None = None,
    help_string: str = "path to config file.",
    *,
    flag_values: flags.FlagValues = flags.FLAGS,
    lock_config: bool = False,
) -> flags.FlagHolder[configdict_base.ConfigDict]:
  """Defines flag for `ConfigDict`.

  This is a wrapper over `ml_collections.config_flags.DEFINE_config_file` which
  set the relevant default params.

  Args:
    name: Flag name, optionally including extra config after a colon.
    default: Default value of the flag (default: None).
    help_string: Help string to display when --helpfull is called. (default:
      "path to config file.")
    flag_values: FlagValues instance used for parsing. (default:
      absl.flags.FLAGS)
    lock_config: If set to True, loaded config will be locked through calling
      .lock() method on its instance (if it exists). (default: True)

  Returns:
    a handle to defined flag.
  """
  return config_flags.DEFINE_config_file(
      name,
      default,
      help_string,
      flag_values=flag_values,
      lock_config=lock_config,
      accept_new_attributes=True,
  )
