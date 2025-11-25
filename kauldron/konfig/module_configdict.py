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

import ast
import functools
import re
import types
from typing import Any
from kauldron import konfig
from kauldron import kontext
from kauldron.konfig import configdict_base


class AutoNestedConfigDict(configdict_base.ConfigDict):
  """A ConfigDict where setattr does not raise an error if the key is not found.

  Instead, it creates a AutoNestedConfigDict at that key and saves the requested
  value in that dict.
  """

  def __getattr__(self, key):
    if key in self:
      return super().__getattr__(key)
    else:
      super().__setattr__(key, AutoNestedConfigDict())
      return super().__getattr__(key)

  def __getitem__(self, key):
    if key in self or key == "__id__":
      return super().__getitem__(key)
    else:
      super().__setitem__(key, AutoNestedConfigDict())
      return super().__getitem__(key)

  def as_flat_dict(self) -> dict[str, Any]:
    flat_dict = {}
    for k, v in self.items():
      if isinstance(v, AutoNestedConfigDict):
        flat_dict.update(
            {f"{k}.{k2}": v2 for k2, v2 in v.as_flat_dict().items()}
        )
      else:
        flat_dict[k] = v
    return flat_dict

  def __repr__(self):
    obj = super().__repr__()
    return f"{type(self).__name__}({obj})"


class ModuleConfigDict(AutoNestedConfigDict):
  """A ConfigDict where the module config is lazily loaded."""

  def __init__(
      self,
      input_dict: dict[str, Any] | None = None,
      module: types.ModuleType = types.ModuleType("module"),
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

  def _validate_attributes(self):
    """Validate inputs and check that config is instantiable."""

    assert hasattr(
        self.module, "get_config"
    ), "Module must have a get_config function."

    if hasattr(self, "__args__") and any(
        isinstance(v, configdict_base.ConfigDict)
        for v in self.__args__.__dict__.values()
    ):
      raise ValueError(
          "ConfigDict overrides are not allowed in class attributes of"
          f" ConfigArgs. Got  {self.__args__}."
      )

  @functools.cached_property
  def config_args(self) -> Any:
    """Get config args from cli_str_arg."""
    if hasattr(self.module, "ConfigArgs"):
      # extract cli args
      class_args = _parse_kv_string_to_dict(self.cli_str_arg)

      # apply overrides
      class_args.update(self.__args__)
      args = self.module.ConfigArgs(**class_args)
      return args
    else:
      return self.cli_str_arg

  @functools.cached_property
  def module_config(self) -> konfig.ConfigDict:
    """Initialize config from config file.

    Returns:
      the built config
    """
    # validate class args
    self._validate_attributes()

    # get args and instantiate config
    try:
      if self.config_args:
        config = self.module.get_config(self.config_args)
      else:
        config = self.module.get_config()
    except Exception as e:
      raise ValueError(
          f"Failed to instantiate config from module {self.module} with args"
          f" {self.config_args}"
      ) from e

    # merge with cfg overrides which are stored in root of self.
    _apply_overrides(config, overrides=self.as_flat_dict())

    return config

  def to_json(self):
    """Build config before serializing to json."""
    return self.read_and_build().to_json()


def _parse_kv_string_to_dict(data_string: str) -> dict[str, Any]:
  """Parses a keyword-argument-like string into a Python dictionary.

  It uses ast.literal_eval for safe evaluation.

  Args:
    data_string: A string like 'key1=(val1,),key2=(val2,)'.

  Returns:
    A dictionary containing the parsed data.
  """

  dict_string = re.sub(
      r"(\w+)=",  # Target: one or more word chars (\w+) followed by '='
      r"'\1':",  # Replacement: Quoted key ('\1') followed by ':'
      data_string,
  )

  # Final step: Enclose in dictionary braces
  dict_string = "{" + dict_string + "}"

  # 2. Safely evaluate the resulting dictionary string
  try:
    result_dict = ast.literal_eval(dict_string)
    return result_dict
  except (ValueError, SyntaxError) as e:
    print(f"Error parsing string: {e}")
    print(f"Attempted to parse: {dict_string}")
    return {}


def _apply_overrides(config: konfig.ConfigDict, overrides: dict[str, Any]):
  """Apply overrides to config."""
  for k, v in overrides.items():
    if not k.startswith("__args__"):
      kontext.set_by_path(config, k, v)


def get_config_from_module(
    module: types.ModuleType, cli_str_arg: str = ""
) -> konfig.ConfigDict:
  """Get config from module and cli_str_arg."""
  modulecfg = ModuleConfigDict(module=module, cli_str_arg=cli_str_arg)
  return modulecfg.module_config
