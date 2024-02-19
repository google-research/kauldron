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

"""End-to-end test."""

import os

from etils import epath
from kauldron import kd
from examples import mnist_autoencoder


def test_end2end(tmp_path: epath.Path):
  # Load config and reduce size
  cfg = mnist_autoencoder.get_config()
  cfg.train_ds.batch_size = 2
  cfg.evals.eval.ds.batch_size = 1  # pytype: disable=attribute-error
  cfg.model.encoder.features = 3
  cfg.num_train_steps = 1
  cfg.workdir = os.fspath(tmp_path)

  trainer = kd.konfig.resolve(cfg)

  # Launch train
  with kd.kmix.testing.mock_data(num_examples=10):
    trainer.train()
