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
from typing import Any

from etils import epath
from etils import epy
from etils import exm
from kauldron import konfig
from kauldron.utils import constants
from kauldron.utils.status_utils import status  # pylint: disable=g-importing-member

from unittest import mock as _mock ; xmanager_api = _mock.Mock()

if typing.TYPE_CHECKING:
  from kauldron.train import trainer_lib  # pylint: disable=g-bad-import-order


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
    from kauldron import kd

    config = kd.konfig.resolve(xp.config)

  xp.root_dir.rmtree()
  ```
  """

  exp: xmanager_api.Experiment
  _: dataclasses.KW_ONLY
  wid: int
  lazy: bool = False

  @classmethod
  def from_xid(cls, xid: int, wid: int, *, lazy: bool = False) -> Experiment:
    """Factory from an xid.

    Args:
      xid: Experiment id
      wid: Work unit id
      lazy: Specify how the imports should be resolved. If `lazy=False`
        (default), the imports are resolved during `xp.config` access. If
        `lazy=True`, the imports are resolved during `konfig.resolve`.

    Returns:
      The experiment object
    """
    return cls(_client().get_experiment(xid), wid=wid, lazy=lazy)

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
  def config(self) -> konfig.ConfigDictLike[trainer_lib.Trainer]:
    """Unresolved `ConfigDict`."""
    config_path = self.wu.workdir / constants.CONFIG_FILENAME
    config = json.loads(config_path.read_text())
    # Wrap the dict to ConfigDict
    return _json_to_config(config, lazy=self.lazy)  # pytype: disable=bad-return-type

  @functools.cached_property
  def trainer(self) -> trainer_lib.Trainer:
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


def _json_to_config(json_value, *, lazy: bool):
  """Wraps a `dict` to a `ConfigDict` and convert list to tuple."""
  match json_value:
    case dict():
      values = {k: _json_to_config(v, lazy=lazy) for k, v in json_value.items()}
      if not lazy and (module_name := _get_module_name(values)):
        try:
          importlib.import_module(module_name)
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
      return [_json_to_config(v, lazy=lazy) for v in json_value]
    case _:
      return json_value


def _get_module_name(values: dict[str, Any]) -> str | None:
  """Returns the module name from the config values."""
  qualname = values.get(konfig.configdict_proxy.QUALNAME_KEY) or values.get(
      konfig.configdict_proxy.CONST_KEY
  )
  if qualname is not None:
    return qualname.split(':', 1)[0]
  else:
    return None


def add_log_artifacts(add_experiment_artifact: bool = False) -> None:
  """Add XManager artifacts for easy access to the Python logs."""
  if not status.on_xmanager or not status.is_lead_host:
    return


def add_colab_artifacts() -> None:
  """Add a link to the kd-infer colab."""
  if not status.on_xmanager or not status.is_lead_host:
    return

  exm.add_work_unit_artifact('https://kauldron.rtfd.io/en/latest-infer (Colab)', _get_kd_infer_url())


def add_tags_to_xm(tags: list[str] | None) -> None:
  """Add tags to the xmanager experiment."""
  if not tags or not status.on_xmanager or not status.is_lead_host:
    return

  assert isinstance(tags, list)
  experiment = exm.current_experiment()
  experiment.add_tags(*tags)


def _get_kd_infer_url() -> str:
  wu = exm.current_work_unit()
  template_params = {
      'XID': wu.experiment_id,
      'WID': wu.id,
  }
  return f'http://https://kauldron.rtfd.io/en/latest-infer#templateParams={json.dumps(template_params)}'
