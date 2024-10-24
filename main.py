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

r"""Training binary.

"""

from __future__ import annotations

import contextlib

import sys
from unittest import mock
# TODO(klausg): remove once we have a better solution
sys.modules['etils.exm'] = mock.MagicMock()
from absl import app
from absl import flags
from etils import epy
import jax
import tensorflow as tf

from kauldron import kd
from kauldron.utils import sweep_utils

_CONFIG = kd.konfig.DEFINE_config_file(
    "cfg",
    None,
    "Training configuration.",
    lock_config=False,
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
  tf.config.set_visible_devices([], "GPU")

  with _wu_error_handling(_POST_MORTEM.value):
    eval_names = _EVAL_NAMES.value
    cfg = sweep_utils.update_with_sweep(
        config=_CONFIG.value,
        sweep_kwargs=_SWEEP_CONFIG.value,
    )
    # TODO(b/374268398): add back _update_xm_configuration(cfg) once ACtx good
    if eval_names is None:
      trainer: kd.train.Trainer = kd.konfig.resolve(cfg)
      trainer.train()
    else:
      # Orbax does not support CheckpointManagers creating the same root
      # directory. By setting `create=False`, we ensure that the checkpoint
      # manager does not create a new root directory.
      if hasattr(cfg, "checkpointer"):
        if hasattr(cfg.checkpointer, "create"):
          cfg.checkpointer.create = False
      trainer: kd.train.Trainer = kd.konfig.resolve(cfg)
      trainer.continuous_eval(eval_names)


@contextlib.contextmanager
def _wu_error_handling(post_mortem: bool = False):
  yield  # not yet supported externally


def _update_xm_configuration(cfg: kd.konfig.ConfigDict) -> None:
  pass  # not supported externally


def _flags_parser(args: list[str]) -> None:
  """Flag parser."""
  # Import everything, except kxm (XManager not included in the trainer binary)
  with (
      epy.binary_adhoc(),
      kd.konfig.set_lazy_imported_modules(
          # This is quite fragile if the user try to import some external
          # XM util/config. In which case user should import this extra module
          # in `with konfig.imports(lazy=True):`
          lazy_import=[
              "kauldron.xm",
              "kauldron.kxm",
              "xmanager",
          ],
      ),
  ):
    flags.FLAGS(args)


if __name__ == "__main__":
  # Adds jax flags to the program.
  jax.config.parse_flags_with_absl()
  app.run(main)  # external
