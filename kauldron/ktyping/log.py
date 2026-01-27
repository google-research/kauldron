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

"""Logging and issue reporting."""

import logging
import warnings
from kauldron.ktyping import config

# 1. Define the top-level logger
logger = logging.getLogger(__name__)
# 2. Add NullHandler immediately
logger.addHandler(logging.NullHandler())


def report(message, policy: config.ReportingPolicy):
  """Reports a message according to the given policy."""
  if policy == config.ReportingPolicy.IGNORE:
    return
  elif policy == config.ReportingPolicy.LOG_INFO:
    logger.info(message, stacklevel=2)
  elif policy == config.ReportingPolicy.WARN:
    warnings.warn(message, stacklevel=2)
  elif policy == config.ReportingPolicy.ERROR:
    raise RuntimeError(message)
