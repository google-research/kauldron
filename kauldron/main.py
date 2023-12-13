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
from kauldron import kd
from kauldron.train.status_utils import status  # pylint: disable=g-importing-member
from kauldron.utils import sweep_utils
from ml_collections import config_flags


_CONFIG = config_flags.DEFINE_config_file(
    "config",
    None,
    "Training configuration.",
    lock_config=False,
    accept_new_attributes=True,
)
_SWEEP_CONFIG = sweep_utils.define_sweep_flag()
_EVAL_NAMES = flags.DEFINE_list(
    "eval_names",
    None,
    "Evaluation(s) to run. When set, run `.continuous_eval()` rather than"
    " `.train()`.",
)
_POST_MORTEM = flags.DEFINE_boolean(
    "catch_post_mortem",
    False,
    "No effect for now.",
)


def main(_):
  with _wu_error_handling(_POST_MORTEM.value):
    eval_names = _EVAL_NAMES.value
    cfg = sweep_utils.update_with_sweep(
        config=_CONFIG.value,
        sweep_kwargs=_SWEEP_CONFIG.value,
    )
    trainer: kd.train.Trainer = kd.konfig.resolve(cfg)
    if eval_names is None:
      trainer.train()
    else:
      trainer.continuous_eval(eval_names)


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


def _flags_parser(args: list[str]) -> None:
  """Flag parser."""
  # Import everything, except kxm (XManager not included in the trainer binary)
  with kd.konfig.set_lazy_imported_modules(
      lazy_import=["kauldron.kxm"],
  ):
    flags.FLAGS(args)


if __name__ == "__main__":
  # Adds jax flags to the program.
  jax.config.parse_flags_with_absl()
  app.run(main)  # external
