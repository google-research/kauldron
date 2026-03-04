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

r"""Simple MNIST classification example using KdNnxModule.

```sh
python -m kauldron.main \
    --cfg=examples/contrib/nnx_mnist.py \
    --cfg.workdir=/tmp/kauldron_oss/workdir
```

"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  from kauldron.contrib.modules import knnx_examples
  import optax
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()
  cfg.seed = 42

  cfg.train_ds = _make_ds(training=True)

  cfg.model = knnx_examples.SimpleKdNnxModule(
      input_dim=28 * 28,
      hdim=128,
      output_dim=10,
  )

  cfg.num_train_steps = 1000

  cfg.train_losses = {
      "xent": kd.losses.SoftmaxCrossEntropyWithIntLabels(
          logits="preds", labels="batch.label"
      ),
  }

  cfg.train_metrics = {
      "accuracy": kd.metrics.Accuracy(logits="preds", labels="batch.label"),
  }

  # activate rng keys for val
  cfg.rng_streams = kd.train.RngStreams(
      [kd.train.RngStream(name="default", init=True, train=True, eval=True)]
  )

  cfg.optimizer = optax.adam(learning_rate=0.003)

  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=500,
  )

  cfg.evals = {
      "eval": kd.evals.Evaluator(
          run=kd.evals.EveryNSteps(100),
          num_batches=None,
          ds=_make_ds(training=False),
      )
  }

  return cfg


def _make_ds(training: bool):
  return kd.data.py.Tfds(
      name="mnist",
      split="train" if training else "test",
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      transforms=[
          kd.data.Elements(keep=["image", "label"]),
          kd.data.ValueRange(key="image", vrange=(0, 1)),
          kd.data.Rearrange(key="image", pattern="... h w c -> ... (h w c)"),
          kd.data.Rearrange(key="label", pattern="... -> ... 1"),
      ],
      batch_size=32,
  )
