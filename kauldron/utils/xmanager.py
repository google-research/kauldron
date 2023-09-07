# Copyright 2023 The kauldron Authors.
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
import json
from typing import Optional

from etils import epath
from kauldron import konfig

from unittest import mock as _mock ; xmanager_api = _mock.Mock()


@dataclasses.dataclass
class RunConfig:
  """Executable config.

  Attributes:
    target: Build target. If not defined, is auto-detected from the config path.
  """

  target: Optional[str] = None


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
  def artifacts(self) -> dict[str, str]:
    """Mapping artifact name -> value."""
    return {a.description: a.artifact for a in self.exp.get_artifacts()}

  @functools.cached_property
  def root_dir(self) -> epath.Path:
    """Root directory of the artifact."""
    return epath.Path(self.artifacts['Workdir'])

  @functools.cached_property
  def config(self) -> konfig.ConfigDict:
    """Unresolved `ConfigDict`."""
    # Use a constant rather than hardcoding `config.json`
    config_path = self.root_dir / str(self.wid) / 'config.json'
    config = json.loads(config_path.read_text())
    config = _json_list_to_tuple(config)
    return konfig.ConfigDict(config)

  @functools.cached_property
  def resolved_config(self) -> konfig.ConfigDict:
    """Resolved `ConfigDict`."""
    return konfig.resolve(self.config)


def _json_list_to_tuple(json_value):
  """Normalize the `json` to use `tuple` rather than `list`."""
  match json_value:
    case dict():
      return {k: _json_list_to_tuple(v) for k, v in json_value.items()}
    case list():
      return tuple(_json_list_to_tuple(v) for v in json_value)
    case _:
      return json_value
