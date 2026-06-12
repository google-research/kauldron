# Copyright 2026 The kauldron Authors.
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
from flax import linen as nn
from flax import struct
import jax.numpy as jnp
from kauldron import kd
from examples import mnist_autoencoder
import numpy as np
import tensorflow_datasets as tfds


def test_end2end(tmp_path: epath.Path):
  # Load config and reduce size
  cfg = mnist_autoencoder.get_config()
  cfg.train_ds.batch_size = 2
  cfg.evals.eval.ds.batch_size = 1  # pytype: disable=attribute-error
  with kd.konfig.mock_modules():
    cfg.model.encoder = nn.Sequential([
        nn.Dense(features=3),
        nn.BatchNorm(use_running_average=False),
    ])
  cfg.num_train_steps = 1
  cfg.workdir = os.fspath(tmp_path)

  trainer = kd.konfig.resolve(cfg)

  # Launch train
  with tfds.testing.mock_data(num_examples=10):
    state, _ = trainer.train()

  assert list(state.collections) == ['batch_stats']


class FilenameSummary(kd.metrics.Metric):
  """Custom summary verifying that real strings survive on the host batch."""

  @struct.dataclass
  class State(kd.metrics.AutoState["FilenameSummary"]):
    dummy: int = 0

    def compute(self, batch_host=None):
      assert batch_host is not None, "batch_host was not passed to compute!"
      assert "filenames" in batch_host, "filenames key missing in batch_host!"
      assert batch_host["filenames"][0].startswith(
          "sample_"
      ), f"Wrong filename: {batch_host['filenames'][0]}"
      return np.array(1.0)

  def get_state(self, **_) -> "FilenameSummary.State":
    return self.State()


def test_host_batch_separation(tmp_path: epath.Path):
  cfg = mnist_autoencoder.get_config()

  with kd.konfig.mock_modules():
    cfg.model.encoder = nn.Sequential([
        nn.Dense(features=3),
        nn.BatchNorm(use_running_average=False),
    ])
    cfg.evals = {}

  cfg.num_train_steps = 1
  cfg.workdir = os.fspath(tmp_path)

  trainer = kd.konfig.resolve(cfg)

  def _load_data():
    return {
        "image": np.ones((10, 28, 28, 1), np.float32),
        "filenames": np.array(
            [f"sample_{i}.png" for i in range(10)], np.dtype("O")
        ),
    }

  pipeline = kd.data.InMemoryPipeline(
      loader=_load_data,
      batch_size=2,
      num_epochs=1,
  )
  trainer = trainer.replace(
      train_ds=pipeline,
      eval_ds=pipeline,
      train_summaries={"test_filenames": FilenameSummary()},
  )

  # Launch train (should succeed without runtime TypeError!)
  trainer.train()
