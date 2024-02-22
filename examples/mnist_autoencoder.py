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

r"""Minimal example training a simple Autoencoder on MNIST.

`--xp.use_interpreter` to launch with `ml_python` (no BUILD rules).

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
  --cfg=third_party/py/kauldron/examples/mnist_autoencoder.py \
  --xp.use_interpreter \
  --xp.platform=jf=2x2
```

"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from flax import linen as nn
  from kauldron import kd
  import optax
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()
  cfg.seed = 42

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

  cfg.train_metrics = {
      "latent_norm": kd.metrics.Norm(tensor="interms.encoder.__call__[0]"),
      "param_norm": kd.metrics.TreeMap(
          metric=kd.metrics.Norm(tensor="params", axis=None)
      ),
      "grad_norm": kd.metrics.TreeReduce(
          metric=kd.metrics.Norm(tensor="grads", axis=None)
      ),
  }

  cfg.train_summaries = {
      "gt": kd.summaries.ShowImages(images="batch.image", num_images=5),
      "recon": kd.summaries.ShowImages(images="preds.image", num_images=5),
  }

  # Optimizer
  cfg.schedules = {}

  cfg.optimizer = optax.adam(learning_rate=0.003)

  # Checkpointer
  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=500,
  )

  cfg.profiler = kd.inspect.Profiler(
      all_host=True,
  )

  cfg.evals = {
      "eval": kd.evals.Evaluator(
          run=kd.evals.RunEvery(100),
          num_batches=None,
          ds=_make_ds(training=False),
          metrics={},
      )
  }

  return cfg


def _make_ds(training: bool):
  return kd.kmix.Tfds(
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
