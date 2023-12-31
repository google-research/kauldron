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

"""End-to-end test."""

import os

from etils import epath
from kauldron import kd
from examples import mnist_autoencoder
import tensorflow_datasets as tfds


def test_end2end(tmp_path: epath.Path):
  # Load config and reduce size
  cfg = mnist_autoencoder.get_config()
  # TODO(epot): Currently the gain mock data is not working:
  # * Require to add `data=` to the BUILD rules (so don't support custom ds)
  # * Mock fail to infer the correct structure for some unknown reason.
  # So instead uses TFDS loader + mock_data for now.
  cfg.train_ds.loader.__qualname__ = 'kauldron.data.loaders.Tfds'
  cfg.train_ds.batch_size = 2
  cfg.evals.eval.ds.loader.__qualname__ = (  # pytype: disable=attribute-error
      'kauldron.data.loaders.Tfds'
  )
  cfg.evals.eval.ds.batch_size = 1  # pytype: disable=attribute-error
  cfg.model.encoder.features = 3
  cfg.num_train_steps = 1
  cfg.workdir = os.fspath(tmp_path)

  cfg = kd.konfig.resolve(cfg)

  # Launch train
  with tfds.testing.mock_data():
    cfg.train()
