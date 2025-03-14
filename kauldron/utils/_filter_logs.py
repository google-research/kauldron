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

"""Helper to filter logs from verbose modules."""
import dataclasses
import logging as std_logging
from absl import logging


@dataclasses.dataclass(kw_only=True, frozen=True)
class _ModuleFilter:
  """Filter out all logs that are INFO or below for the given modules."""
  modules_to_filter: set[str]

  def filter(self, record):
    if record.levelno <= std_logging.INFO:
      return record.module not in self.modules_to_filter
    return True


def add_filter(modules_to_filter: set[str]):
  """Add a filter to the absl logging handler."""
  log_filter = _ModuleFilter(modules_to_filter=modules_to_filter)
  logging.get_absl_handler().addFilter(log_filter)
