# Copyright 2026 The kauldron Authors.
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

"""Configuration for ktyping."""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Iterable
import uuid

from etils import epy
from kauldron.ktyping import utils
from kauldron.ktyping.internal_typing import MISSING, Missing  # pylint: disable=g-importing-member, g-multiple-import


class ReportingPolicy(enum.StrEnum):
  IGNORE = enum.auto()
  LOG_INFO = enum.auto()
  WARN = enum.auto()
  ERROR = enum.auto()


# MARK: Config
@dataclasses.dataclass(kw_only=True, frozen=False)
class Config(epy.ContextManager):
  """Configuration for ktyping."""

  # Enable/disable all typechecked decorators and custom typeguard checkers.
  typechecking_enabled: bool | Missing = MISSING

  # What to do about jaxtyping annotations mixed into ktyping scopes.
  jaxtyping_annotations: ReportingPolicy | Missing = MISSING

  # ----------------------------------------------------------------------------
  replace = dataclasses.replace

  def update(self, other: Config) -> Config:
    """Returns a new Config that is the merge of the two configs."""
    replace_fields = {}
    for field in dataclasses.fields(Config):
      if getattr(other, field.name) is not MISSING:
        replace_fields[field.name] = getattr(other, field.name)

    return self.replace(**replace_fields)

  def __contextmanager__(self) -> Iterable[Config]:
    """Context manager to temporarily override the config.

    Example usage:
    ```
    with kt.config.Config(typechecking_enabled=False):
      # do something
    ```

    Yields:
      The current config.
    """
    cfg_id = add_config_override(r".*", self)
    yield self
    remove_config_override(cfg_id)


# MARK: Global config
# Global configuration for ktyping
CONFIG = Config(
    typechecking_enabled=True,
    jaxtyping_annotations=ReportingPolicy.ERROR,
)

# List of config overrides for specific modules (by module name regex).
CONFIG_OVERRIDES: dict[uuid.UUID, tuple[str, Config]] = {}
# Example:
#   kt.add_config_override(
#     (r"example\.project", Config(typechecking_enabled=False)
#   )


def add_config_override(module_regex: str, config: Config) -> uuid.UUID:
  """Adds a config override for a given module regex.

  Args:
    module_regex: A regex that matches the module name of the source code. This
      is match against the module name of typecked function or class with
      `re.match` (so the match is anchored at the start but not the end of the
      module name)
    config: The config to override the default config with. Only fields that are
      not left as `MISSING` will be overridden.

  Returns:
    The ID of the config override. This can be used to remove the override
    using `kt.remove_config_override`.
  """
  cfg_id = uuid.uuid4()
  CONFIG_OVERRIDES[cfg_id] = (module_regex, config)
  return cfg_id


def remove_config_override(cfg_id: uuid.UUID, missing_ok: bool = False) -> None:
  """Removes a config override at the given index.

  Args:
    cfg_id: The ID of the config override to remove as returned by
      `kt.add_config_override`.
    missing_ok: If True, do not raise an error if the config override is not
      found.

  Raises:
    KeyError: If the config override is not found and `missing_ok` is False.
  """
  if cfg_id in CONFIG_OVERRIDES:
    del CONFIG_OVERRIDES[cfg_id]
  elif not missing_ok:
    raise KeyError(f"Config override {cfg_id} not found.")


# MARK: get_config
def get_config(source: utils.CodeLocation | None = None) -> Config:
  """Returns the config entry for a given source code location."""
  config = CONFIG

  # Search through config overrides by module name.
  module_name = source.module_name if source else "<unknown>"
  for module_regex, config_override in CONFIG_OVERRIDES.values():
    if re.match(module_regex, module_name):
      config = config.update(config_override)

  return config
