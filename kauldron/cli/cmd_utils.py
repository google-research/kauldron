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

"""Base class for all sub-commands."""

from __future__ import annotations

import abc
import contextlib
import dataclasses
import functools
import time
from typing import Any, Iterator

from etils import enp
from etils import epy
import jax
from kauldron import konfig
from kauldron import kontext
from kauldron.cli import patch_config
from kauldron.utils import immutabledict


@dataclasses.dataclass(frozen=True, kw_only=True)
class CommandGroup:
  """Base class for all commands.

  These are only containers to group sub-commands in the CLI and can be used to
  define shared flags (e.g. --datasets, --eval_mode).

  These flags are automatically registered by simple_parsing from the
  attributes of the class.

  Subclasses of this class need to override the `sub_command` attribute, which
  and annotate it with a Union of the desired sub-commands to make them
  available as positional CLI commands.
  """

  sub_command: SubCommand


@dataclasses.dataclass(frozen=True, kw_only=True)
class SubCommand(abc.ABC):
  """Base class for all sub-commands.

  Subclasses of this class need to be dataclasses and implement `__call__`.

  Additional CLI flags can be defined as attributes and will be automatically
  be picked up and registered by simple_parsing.
  """

  # `cmd=False` prevents simple_parsing from registering --cfg as a CLI arg,
  # which would conflict with the absl --cfg flag defined in main.py.
  cfg: konfig.ConfigDict | None = dataclasses.field(
      default=None, metadata={"cmd": False}
  )
  origin: patch_config.ConfigOrigin | None = dataclasses.field(
      default=None, metadata={"cmd": False}
  )
  group: CommandGroup | None = dataclasses.field(
      default=None, metadata={"cmd": False}
  )

  @functools.cached_property
  def trainer(self):
    with timed("Resolving config"):
      return konfig.resolve(self.cfg, freeze=False)

  def print_config_origin(self):
    if self.origin is not None:
      print(self.origin.summary())

  @abc.abstractmethod
  def __call__(self) -> None:
    """Executes the command."""


def execute_command(
    group: CommandGroup,
    cfg: konfig.ConfigDict,
    origin: patch_config.ConfigOrigin,
) -> None:
  """Executes the command and pretty-print the result."""
  cmd = dataclasses.replace(
      group.sub_command,
      cfg=cfg,
      origin=origin,
      group=group,
  )
  cmd()


def tracked_update(
    cfg: konfig.ConfigDict, path: str, value: Any
) -> dict[str, Any]:
  """Sets value at path and returns {concrete_path: value} for tracking."""
  modified = kontext.set_by_path(cfg, path, value)
  return dict.fromkeys(modified, value)


@contextlib.contextmanager
def timed(action: str) -> Iterator[None]:
  start = time.time()
  print(f"{action}...", end="", flush=True)
  yield
  print(f"done ({time.time() - start:.1f} sec)")


def print_spec(obj):
  """Print the state spec."""
  obj_spec = jax.tree.map(
      lambda x: enp.ArraySpec.from_array(x)
      if isinstance(x, jax.ShapeDtypeStruct) or enp.is_array(x)
      else x,
      obj,
  )
  obj_spec = immutabledict.unfreeze(obj_spec)
  epy.pprint(obj_spec)
