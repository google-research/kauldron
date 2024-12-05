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

"""Status utils is a small library to reduce boilerplate of common check.

```python
from kauldron.utils.status_utils import status


status.log_status("Starting training.")
if status.on_xmanager and status.is_lead_host:
  ...
```
"""

import functools
import warnings

from absl import logging
from etils import epy
import jax

from unittest import mock as _mock ; xmanager_api = _mock.Mock()


class _Status:
  """Reduce boilerplate for common operations.

  Example:

  ```python
  from kauldron.utils.status_utils import status


  status.log_status("Starting training.")
  if status.on_xmanager and status.is_lead_host:
    ...
  ```
  """

  @functools.cached_property
  def on_xmanager(self) -> bool:
    return False

  @functools.cached_property
  def is_lead_host(self) -> bool:
    return jax.process_index() == 0

  @functools.cached_property
  def xp(self) -> xmanager_api.Experiment:
    if not self.on_xmanager:
      raise RuntimeError("Not running on Xmanager.")

    return xmanager_api.XManagerApi().get_current_experiment()

  @functools.cached_property
  def wu(self) -> xmanager_api.WorkUnit:
    if not self.on_xmanager:
      raise RuntimeError("Not running on Xmanager.")

    return xmanager_api.XManagerApi().get_current_work_unit()

  @functools.cached_property
  def wid(self) -> int:
    if not self.on_xmanager:
      raise RuntimeError("Not running on Xmanager.")
    return self.wu.id

  @functools.cached_property
  def xid(self) -> int:
    if not self.on_xmanager:
      raise RuntimeError("Not running on Xmanager.")
    return self.wu.experiment_id

  def log_status(self, msg: str) -> None:
    """Log a message (from lead host), will be displayed on the XM UI."""
    self.log(msg, stacklevel=2)
    if self.on_xmanager and self.is_lead_host:
      self.wu.set_notes(msg)

  def log(self, msg: str, *, stacklevel: int = 1) -> None:
    """Print a message.

    * On Colab: Use `print`
    * On Borg: Use `logging.info`

    Contrary to `log_status()`, messages are not displayed on the XM UI.

    Args:
      msg: the message to print.
      stacklevel: If wrapping this function, indicate the number of frame to
        skip so logging display the correct caller site.
    """
    if (
        not epy.is_notebook()
    ):
      logging.info(msg, stacklevel=1 + stacklevel)
      return
    else:
      print(msg, flush=True)  # Colab or local

  def warn(
      self,
      msg: str,
      category: type[Warning] | None = None,
      *,
      stacklevel: int = 1,
  ) -> None:
    """Print a warning."""
    warnings.warn(msg, category, stacklevel=1 + stacklevel)
    if epy.is_notebook():
      category = category or Warning
      print(f"{category.__name__}: {msg}")


status = _Status()
