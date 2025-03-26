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

r"""Small example for a standalone evaluator using the ReadoutWrapper module.

It trains a simple dense readout head for MNIST classification on an arbitrary
experiment. This file assumes an xid of a `examples/mnist_autencoder.py` run.
But this can be adapted to different models by adjusting the parameters of the
`ReadoutWrapper` module:
  - cfg.model.model_inputs
  - cfg.model.readout_inputs

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
  --cfg=third_party/py/kauldron/examples/contrib/readout_mnist.py \
  --xp.use_interpreter \
  --xp.platform=jf=2x2 \
  --cfg.aux.xid=96522893 \
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

  cfg.aux = {"xid": konfig.placeholder(int)}  # NOTE: set from the commandline

  # Dataset
  cfg.train_ds = _make_ds(training=True)

  # Model
  cfg.model = kd.contrib.nn.ReadoutWrapper(
      # Inner model
      model_inputs={"inputs": "batch.image"},
      model=kd.contrib.nn.get_model_from_xid(xid=cfg.ref.aux.xid),
      # Readout heads
      readout_inputs={
          "classification": {"inputs": "interms.model.encoder.__call__[0]"}
      },
      readout_heads={
          "classification": nn.Dense(features=10, name="readout_head")
      },
      finetune=False,
  )

  cfg.init_transform = kd.ckpts.PartialKauldronLoader(
      workdir=kd.ckpts.workdir_from_xid(xid=cfg.ref.aux.xid),
      new_to_old={  # Mapping params
          "params.model": "params",
      },
  )

  # Training
  cfg.num_train_steps = 1000

  # Losses
  cfg.train_losses = {
      "xent": kd.losses.SoftmaxCrossEntropyWithIntLabels(
          logits="preds.readouts.classification", labels="batch.label"
      ),
  }

  cfg.train_metrics = {
      "accuracy": kd.metrics.Accuracy(
          logits="preds.readouts.classification", labels="batch.label"
      ),
  }
  # Optimizer
  cfg.optimizer = optax.adam(learning_rate=0.003)

  # Checkpointer
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
  return kd.data.tf.Tfds(
      name="mnist",
      split="train" if training else "test",
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      transforms=[
          kd.data.Elements(keep=["image", "label"]),
          kd.data.ValueRange(key="image", vrange=(0, 1)),
          kd.data.Rearrange(key="label", pattern="... -> ... 1"),
      ],
      batch_size=256,
  )
