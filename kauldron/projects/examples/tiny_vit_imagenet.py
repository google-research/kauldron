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

"""Runs a ViT-Tiny classifer on Imagenet 64x64."""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  import optax
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Config()
  cfg.workdir = konfig.placeholder(str)  # will be set by the launcher
  cfg.seed = 42

  # Dataset
  cfg.train_ds = _make_ds(training=True)

  # Model
  cfg.model = kd.nn.Vit(
      image="batch.image",
      encoder=kd.nn.VitEncoder.from_variant_str("ViT-Ti/8"),
      num_classes=1000,
  )

  # Training
  cfg.num_train_steps = 10**6

  # Losses
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
      "roc_auc": kd.metrics.RocAuc(logits="preds.logits", labels="batch.label"),
      "final_attention_std": kd.metrics.Std(
          values="interms.encoder.layers_11.attention.attn_weights[0]"
      ),
  }

  # Optimizer
  cfg.schedules = {
      "learning_rate": optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=0.0003,
          warmup_steps=5_000,
          decay_steps=cfg.ref.num_train_steps,
      )
  }

  cfg.optimizer = optax.adam(learning_rate=cfg.ref.schedules["learning_rate"])

  # Checkpointer
  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=1000,
      max_to_keep=1,
  )

  cfg.profiler = kd.inspect.Profiler(
      all_host=True,
  )

  cfg.evals = {
      "eval": kd.train.Evaluator(
          run_every=1000,
          num_batches=10,
          ds=_make_ds(training=False),
      )
  }
  return cfg


def _make_ds(training: bool):
  return kd.data.TFDataPipeline(
      loader=kd.data.loaders.GrainTfds(
          name="imagenet_resized/64x64",
          split="train" if training else "validation",
          shuffle=True if training else False,
          num_epochs=None if training else 1,
      ),
      transformations=[
          kd.data.Elements(keep=["image", "label"]),
          kd.data.ValueRange(key="image", in_vrange=(0, 255), vrange=(0, 1)),
          kd.data.Rearrange(key="label", pattern="... -> ... 1"),
      ],
      batch_size=512,
  )


def sweep():
  for i in range(3):
    for lr in [0.001, 0.003, 0.01]:
      yield {"schedules.learning_rate.peak_value": lr, "seed": i}
