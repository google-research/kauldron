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

r"""Example showcasing the TrainEvaluator which regularly trains a readout.

This config trains an MNIST autoencoder and every 100 steps the
`cfg.evals.readout` evaluator trains a readout head for MNIST classification
based on the `kauldron.examples.contrib.readout_mnist` config.

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
  --cfg=third_party/py/kauldron/examples/contrib/train_evaluator.py \
  --xp.use_interpreter \
  --xp.platform=jf=2x2
```

"""

from kauldron import konfig
# NOTE: import readout config outside of konfig.imports
from examples.contrib import readout_mnist

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from flax import linen as nn
  from kauldron import kd
  import optax
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()

  # Dataset
  cfg.train_ds = _make_ds(training=True)

  # Model
  cfg.model = kd.nn.FlatAutoencoder(
      inputs="batch.image",
      encoder=nn.Dense(features=128),
      decoder=nn.Dense(features=28 * 28),
  )

  # Training
  cfg.num_train_steps = 1000

  # Losses
  cfg.train_losses = {
      "recon": kd.losses.L2(preds="preds.image", targets="batch.image"),
  }

  cfg.train_metrics = {}

  cfg.train_summaries = {
      "gt": kd.summaries.ShowImages(images="batch.image", num_images=5),
      "recon": kd.summaries.ShowImages(images="preds.image", num_images=5),
  }

  # Optimizer
  cfg.optimizer = optax.adam(learning_rate=0.003)

  # Checkpointer
  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=500,
  )

  cfg.evals = {
      "readout": kd.contrib.evals.TrainEvaluator(
          run=kd.evals.EveryNSteps(100),
          readout_config=readout_mnist.get_config(),
      ),
  }

  return cfg


def _make_ds(training: bool):
  return kd.data.tf.Tfds(
      name="mnist",
      split="train" if training else "test",
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      transforms=[
          kd.data.Elements(keep=["image"]),
          kd.data.ValueRange(key="image", vrange=(0, 1)),
      ],
      batch_size=256,
  )
