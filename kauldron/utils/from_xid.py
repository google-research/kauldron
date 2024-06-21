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

"""Helper for loading configs,... from XManager experiments."""

from __future__ import annotations

import contextlib
import json
import typing
from typing import Any, Callable, ContextManager, Optional

from etils import enp
from etils import epath
from kauldron import konfig
from kauldron import kontext
from kauldron.data import utils as data_utils
from kauldron.typing import PyTree  # pylint: disable=g-importing-member
from kauldron.utils import constants
from kauldron.utils import xmanager as xm_lib

if typing.TYPE_CHECKING:
  from kauldron import kd  # pylint: disable=g-bad-import-order


def get_cfg(
    xid: int,
    wid: int = 1,
    *,
    lazy: bool = True,
) -> kd.train.Trainer:
  """Returns the config/sub-config from an xmanager experiment.

  Usage:

  ```python
  cfg = kd.from_xid.get_cfg(12345)
  ```

  Args:
    xid: The XID of the experiment to load the config from.
    wid: ID of the worker to load the config from. Defaults to 1.
    lazy: When `False`, importing the config will also trigger the imports of
      all defined symbols.

  Returns:
    The restored `konfig.ConfigDict`.
  """
  return xm_lib.Experiment.from_xid(xid, wid, lazy=lazy).config


def get_resolved(
    xid: int,
    wid: int = 1,
    *,
    path: str | None = None,
    adhoc_from: None | str | int = None,
    # TODO(epot): Can we remove `config_update_fn` ?
    config_update_fn: Callable[[Any], Any] = lambda x: x,
    # TODO(epot): Add `default=` if `path` isn't found.
) -> Any:
  """Returns the resolved config/sub-config from an xmanager experiment.

  Note: By default, the config from XManager is resolved using the current
  version of the code (not the same as used in the XM experiment). You
  can change this behavior by passing `adhoc_from=`, but only modules
  not yet in `sys.modules` will be adhoc imported.

  If you only care about a sub-config, use `path=` to avoid resolving and
  depending on the whole config.

  Args:
    xid: The XID of the experiment to load the config from.
    wid: ID of the worker to load the config from. Defaults to 1.
    path: Path of the sub-config to resolve (e.g. `model`).
    adhoc_from: If additional dependencies are needed to load the model, specify
      the source to adhoc import from. If adhoc_from is an int, it is assumed to
      be an XID and adhoc_from is set to f"xid/{adhoc_from}".
    config_update_fn: An optional function to update the config before resolving
      the model. Defaults to the identity function.

  Returns:
    The resolved `kd.train.Trainer` (or any other sub-field).
  """
  with _adhoc_cm(adhoc_from):
    cfg = get_cfg(xid, wid)
    if path is not None:
      cfg = kontext.get_by_path(cfg, path)

    cfg = config_update_fn(cfg)
    return konfig.resolve(cfg)


def get_workdir(
    xid: int,
    wid: int = 1,
) -> epath.Path:
  """Returns the workdir of an xmanager experiment."""
  return xm_lib.Experiment.from_xid(xid, wid, lazy=True).wu.workdir


def get_element_spec(
    xid: int,
    wid: int = 1,
) -> PyTree[enp.ArraySpec]:
  """Returns the element_spec of the train dataset of an xmanager experiment.

  This allow to initialize an existing model without having to load an actual
  dataset, by using `kd.data.ElementSpecDataset`.

  Args:
    xid: The XID of the experiment to load the config from.
    wid: ID of the worker to load the config from. Defaults to 1.

  Returns:
    The element_spec of the train dataset.
  """
  path = get_workdir(xid, wid) / constants.ELEMENT_SPEC_FILENAME
  spec = json.loads(path.read_text())
  return data_utils.json_to_spec(spec)


def _adhoc_cm(
    adhoc_from: Optional[str | int] = None,
) -> ContextManager[Any]:
  if isinstance(adhoc_from, int):
    adhoc_from = f"xid/{adhoc_from}"

  if adhoc_from is None:
    return contextlib.nullcontext()
  from etils import ecolab  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

  return ecolab.adhoc(source=adhoc_from, invalidate=False)
