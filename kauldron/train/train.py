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
from kauldron import konfig
from kauldron.train import config_lib
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member
from kauldron.utils import sweep_utils
from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False
)
_SWEEP_CONFIG = sweep_utils.define_sweep_flag()
_POST_MORTEM = flags.DEFINE_boolean(
    "catch_post_mortem",
    False,
    "No effect for now.",
)


def main(_):
  with _wu_error_handling(_POST_MORTEM.value):
    cfg = sweep_utils.update_with_sweep(
        config=_CONFIG.value,
        sweep_kwargs=_SWEEP_CONFIG.value,
    )
    cfg: config_lib.Trainer = konfig.resolve(cfg)
    cfg.train()


@contextlib.contextmanager
def _wu_error_handling(post_mortem: bool = False):
  """Catch and log error."""
  context = contextlib.nullcontext
  with context():
    try:
      yield
    except Exception as e:
      exc_name = type(e).__name__
      status.log(f"🚨 {exc_name}: {e!s}")
      status.xp.add_tags(f"🚨 {exc_name} 🚨")
      raise


if __name__ == "__main__":
  # Adds jax flags to the program.
  jax.config.parse_flags_with_absl()
  app.run(main)  # external
