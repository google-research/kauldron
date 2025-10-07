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

"""ConfigDict extension that can lazily load a module."""

import types
from typing import Any
from kauldron import konfig
from kauldron import kontext
from kauldron.konfig import configdict_base


class FlexConfigDict(configdict_base.ConfigDict):
  """A ConfigDict where setattr does not raise an error if the key is not found.

  Instead, it creates a FlexConfigDict at that key and saves the requested value
  in that dict.
  """

  def __getattr__(self, key):
    if key in self:
      return super().__getattr__(key)
    else:
      super().__setattr__(key, FlexConfigDict())
      return super().__getattr__(key)

  def __getitem__(self, key):
    if key in self or key == "__id__":
      return super().__getitem__(key)
    else:
      super().__setitem__(key, FlexConfigDict())
      return super().__getitem__(key)


class ModuleConfigDict(FlexConfigDict):
  """A ConfigDict where the module config is lazily loaded."""

  def __init__(
      self,
      input_dict: dict[str, Any] = None,
      module: types.ModuleType = None,
      cli_str_arg: str = "",
  ):
    assert (
        module is not None
    ), "A module is required to initialize ModuleConfigDict."
    # use object.__setattr__ since the configdict also uses this functionality
    object.__setattr__(self, "module", module)
    object.__setattr__(self, "cli_str_arg", cli_str_arg)

    super().__init__(input_dict)

  def __repr__(self):
    return (
        f'ModuleConfigDict(module="{self.module}",'
        f' cli_str_arg="{self.cli_str_arg}")\ndict={super().__repr__()}'
    )

  def _validate_class_args(self, class_overrides):
    if any(
        isinstance(v, configdict_base.ConfigDict)
        for v in class_overrides.__dict__.values()
    ):
      raise ValueError(
          "ConfigDict overrides are not allowed in class attributes of Config."
          f" Got  {class_overrides}."
      )

  def read_and_build(self) -> konfig.ConfigDict:
    """Initialize config from config file.

    Returns:
      the built config
    """
    class_overrides = self.__class_args__
    del self.__class_args__
    cfg_overrides = self

    # adhoc imports here
    config_module = self.module

    self._validate_class_args(class_overrides)

    if self.cli_str_arg and hasattr(config_module, self.cli_str_arg):
      # corresponds to  --cfg="path/to/config.py:MyConfigClass"
      config_class = getattr(config_module, self.cli_str_arg)
      config = config_class(**dict(class_overrides)).build()

    elif self.cli_str_arg and hasattr(
        config_module, self.cli_str_arg.split("(")[0]
    ):
      # corresponds to
      # --cfg="path/to/to/config.py:MyConfigClass(arg1=val1,arg2=val2,...)"
      class_name, class_arg = self.cli_str_arg[:-1].split("(", 1)
      class_kwargs = {
          k.split("=")[0]: eval(k.split("=")[1]) for k in class_arg.split(",")
      }
      # TODO(geco): can we do something better than eval, e.g with parsers ?

      class_kwargs.update(dict(class_overrides))

      config = getattr(config_module, class_name)(**class_kwargs).build()

    elif hasattr(config_module, "Config") and hasattr(
        config_module.Config, "build"
    ):
      # corresponds to --cfg="path/to/config.py"
      self._validate_class_args(class_overrides)

      config = config_module.Config(**dict(class_overrides)).build()

    elif hasattr(config_module, "get_config"):
      # corresponds to --cfg="path/to/config.py" and config does not have a
      # Config class
      if self.cli_str_arg:
        config = config_module.get_config(self.cli_str_arg)
      else:
        config = config_module.get_config()

    else:
      raise ValueError(
          "Unable to parse config file. you should declare either a Config"
          " class or a get_config function."
      )

    # merge with cfg overrides
    for k, v in kontext.flatten_with_path(cfg_overrides).items():
      kontext.set_by_path(config, k, v)

    return config

  def to_json(self):
    """Build config before serializing to json."""
    return self.read_and_build().to_json()
