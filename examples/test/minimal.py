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

r"""Temporary test file to debug the open source release.

Run:
  python main.py --cfg=examples/test/minimal.py \
      --cfg.workdir=/tmp/kauldron_oss/workdir

TODO(klausg): remove this file once the other examples are working.
"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  import numpy as np
  import functools
  import optax
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()
  cfg.seed = 42

  # Dataset
  cfg.train_ds = kd.data.InMemoryPipeline(
      loader=functools.partial(np.ones, (1, 1)), batch_size=1
  )

  # Model
  cfg.model = kd.nn.DummyModel()

  # Training
  cfg.num_train_steps = 0

  # Losses
  cfg.train_losses = {}

  cfg.train_metrics = {}

  cfg.train_summaries = {}

  cfg.writer = kd.train.metric_writer.NoopWriter()

  # Optimizer
  cfg.schedules = {}

  cfg.optimizer = optax.sgd(0.1)

  cfg.evals = {}

  cfg.setup = kd.train.setup_utils.Setup(  # pytype: disable=wrong-arg-types
      add_flatboard=False, flatboard_build_context=None
  )

  return cfg
