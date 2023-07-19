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

r"""Training binary.

"""

from __future__ import annotations

import contextlib

from absl import app
from absl import flags
import jax
from kauldron.train import train_lib
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)


def main(_):
  train_lib.train(_CONFIG.value)

if __name__ == "__main__":
  # Adds jax flags to the program.
  jax.config.parse_flags_with_absl()
  app.run(main)
