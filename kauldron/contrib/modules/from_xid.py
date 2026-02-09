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

"""Helper for loading a model (not weights) from an Xmanager experiment."""

from __future__ import annotations

from typing import Any, Callable, Optional

import flax.linen as nn
from kauldron import kd


def get_model_from_xid(
    xid: int,
    wid: int = 1,
    *,
    adhoc_from: Optional[str | int] = None,
    model_path: str | None = "model",
    config_update_fn: Callable[[Any], Any] = lambda x: x,
) -> nn.Module:
  """Returns the model from an xmanager experiment.

  Usage:
  ```
  cfg.model = kd.contrib.nn.get_model_from_xid(12345)
  ```

  Warning: Resolves the config from the xmanager experiment using the current
  version of the code (not the same as used in the experiment).

  Warning: Note that this only works if all the additional dependencies of the
  target experiment config are also imported in the current config.

  Args:
    xid: The XID of the experiment to load the config from.
    wid: ID of the worker to load the config from. Defaults to 1.
    adhoc_from: If additional dependencies are needed to load the model, specify
      the source to adhoc import from. If adhoc_from is an int, it is assumed to
      be an XID and adhoc_from is set to f"xid/{adhoc_from}".
    model_path: Path to the model in the experiment config.
    config_update_fn: An optional function to update the config before resolving
      the model. Defaults to the identity function.

  Returns:
    The `nn.Module` corresponding to the `cfg.model` of the experiment.
  """
  return kd.from_xid.get_resolved(
      xid,
      wid,
      path=model_path,
      adhoc_from=adhoc_from,
      config_update_fn=config_update_fn,
  )
