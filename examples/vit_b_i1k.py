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

r"""Runs a ViT-B/16 classifer on Imagenet 64x64.

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
  --cfg=third_party/py/kauldron/examples/vit_b_i1k.py \
  --xp.use_interpreter \
  --xp.platform=df=2x2
```

"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
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
  # TODO(klausg): classifier token
  cfg.model = kd.nn.Vit(
      image="batch.image",
      encoder=kd.nn.VitEncoder.from_variant_str(
          "ViT-B/16",
          block_type=kd.nn.ParallelAttentionBlock,
          prepend_cls_token=True,
      ),
      num_classes=1000,
      mode="cls_token",
  )

  # Training
  cfg.num_train_steps = 90_000  # approx 300 epochs

  # Losses
  # TODO(klausg): use sigmoid_xent instead?
  cfg.train_losses = {
      "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
          logits="preds.logits",
          labels="batch.label",
      ),
  }

  # Metrics
  cfg.train_metrics = {
      "accuracy": kd.metrics.Accuracy(
          logits="preds.logits", labels="batch.label"
      ),
  }

  # Optimizer
  cfg.schedules = {
      "learning_rate": optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=0.001,
          warmup_steps=10_000,
          decay_steps=cfg.ref.num_train_steps,
      )
  }

  cfg.optimizer = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.adamw(
          learning_rate=cfg.ref.schedules.learning_rate, weight_decay=0.0001
      ),
  )

  # Checkpointer
  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=1000,
      max_to_keep=1,
  )

  cfg.evals = {
      "eval": kd.evals.Evaluator(
          run=kd.evals.RunEvery(2500),
          num_batches=2,
          ds=_make_ds(training=False),
      )
  }
  return cfg


def _make_ds(training: bool):
  """Create a data pipeline for train/eval."""
  if training:
    transformations = [
        kd.data.Elements(keep=["image", "label"]),
        kd.data.InceptionCrop(key="image", resize_size=(224, 224)),
        kd.data.RandomFlipLeftRight(key="image"),
        kd.contrib.data.RandAugment(
            image_key="image", num_layers=2, magnitude=15
        ),
        kd.data.ValueRange(key="image", in_vrange=(0, 255), vrange=(0, 1)),
        kd.data.Rearrange(key="label", pattern="... -> ... 1"),
    ]
  else:
    transformations = [
        kd.data.Elements(keep=["image", "label"]),
        kd.data.ValueRange(key="image", in_vrange=(0, 255), vrange=(0, 1)),
        kd.data.ResizeSmall(key="image", smaller_size=256),
        kd.data.CenterCrop(key="image", shape=(224, 224, 3)),
        kd.data.Rearrange(key="label", pattern="... -> ... 1"),
    ]

  return kd.kmix.Tfds(
      name="imagenet2012",
      split="train[:99%]"
      if training
      else "train[99%:]",  # TODO(klausg) pad instead of drop remainder
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      cache=True,
      transforms=transformations,
      batch_size=4096,
  )


def sweep():
  for i in range(3):
    yield {"seed": i}
