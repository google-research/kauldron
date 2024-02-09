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

"""XManager utils."""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import dataclasses
import functools
import importlib
import json
import typing

from etils import epath
from etils import epy
from kauldron import konfig

from unittest import mock as _mock ; xmanager_api = _mock.Mock()

if typing.TYPE_CHECKING:
  from kauldron.train import config_lib  # pylint: disable=g-bad-import-order


@functools.cache
def _client() -> xmanager_api.XManagerApi:
  return xmanager_api.XManagerApi(xm_deployment_env='alphabet')


@dataclasses.dataclass(frozen=True)
class Experiment:
  """XManager experiment wrapper.

  Usage:

  ```python
  xp = kd.xm.Experiment.from_xid(1234)

  with xp.adhoc():  # Import from xid
    import kauldron as kd

    config = kd.konfig.resolve(xp.config)

  xp.root_dir.rmtree()
  ```
  """

  exp: xmanager_api.Experiment
  _: dataclasses.KW_ONLY
  wid: int

  @classmethod
  def from_xid(cls, xid: int, wid: int) -> Experiment:
    """Factory from an xid.

    Args:
      xid: Experiment id
      wid: Work unit id

    Returns:
      The experiment object
    """
    return cls(_client().get_experiment(xid), wid=wid)

  def __post_init__(self):
    # Ensure `wid` is an `int`.
    object.__setattr__(self, 'wid', int(self.wid))

  @contextlib.contextmanager
  def adhoc(self) -> Iterator[None]:
    """Adhoc imports from the experiment snapshot."""
    # TODO(epot): `kd.xm` is already in kauldron. Should support reloading
    # kauldron but not reload `XManager`

    from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    with ecolab.adhoc(
        f'xid/{self.exp.id}',
        reload='kauldron',
        restrict_reload=False,
        invalidate=False,
    ):
      yield

  @functools.cached_property
  def wu(self) -> WorkUnit:
    return WorkUnit(wu=self.exp.get_work_unit(self.wid))

  @functools.cached_property
  def artifacts(self) -> dict[str, str]:
    """Mapping artifact name -> value."""
    return {a.description: a.artifact for a in self.exp.get_artifacts()}

  @functools.cached_property
  def root_dir(self) -> epath.Path:
    """Root directory of the artifact."""
    return _normalize_workdir(self.artifacts['Workdir'])

  @functools.cached_property
  def config(self) -> konfig.ConfigDictLike[config_lib.Trainer]:
    """Unresolved `ConfigDict`."""
    # Should use a constant rather than hardcoding `config.json`
    config_path = self.wu.workdir / 'config.json'
    config = json.loads(config_path.read_text())
    return _json_to_config(config)  # Wrap the dict to ConfigDict  # pytype: disable=bad-return-type

  @functools.cached_property
  def trainer(self) -> config_lib.Trainer:
    """Resolved `ConfigDict`."""
    return konfig.resolve(self.config)


@dataclasses.dataclass(frozen=True)
class WorkUnit:
  """XManager work unit wrapper."""

  wu: xmanager_api.WorkUnit

  @functools.cached_property
  def artifacts(self) -> dict[str, str]:
    """Mapping artifact name -> value."""
    return {a.description: a.artifact for a in self.wu.get_artifacts()}

  @functools.cached_property
  def workdir(self) -> epath.Path:
    """Root directory of the artifact."""
    return _normalize_workdir(self.artifacts['Workdir'])


def _normalize_workdir(path: str) -> epath.Path:
  """Normalize workdir path."""
  return epath.Path(path)


def _json_to_config(json_value):
  """Wraps a `dict` to a `ConfigDict` and convert list to tuple."""
  match json_value:
    case dict():
      values = {k: _json_to_config(v) for k, v in json_value.items()}
      if qualname := values.get('__qualname__'):
        try:
          importlib.import_module(qualname.split(':', 1)[0])
        except ImportError as e:
          epy.reraise(
              e,
              suffix=(
                  '\nOn Colab, you might need to access the config from a adhoc'
                  ' import context.'
              ),
          )
      return konfig.ConfigDict(values)
    case list():
      return tuple(_json_to_config(v) for v in json_value)
    case _:
      return json_value
