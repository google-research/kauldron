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

r"""Runs a ViT-Tiny classifer on Imagenet 64x64.

This config showcases some advanced features. It is not meant to be forked
as-is but serve as inspiration to pick & choose from.

```sh
python -m kauldron.main \
    --cfg=examples/tiny_vit_advanced.py \
    --cfg.workdir=/tmp/kauldron_oss/workdir
```
"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from jax.experimental import checkify
  from kauldron import kd
  from kauldron import kxm
  import optax
  from xmanager import xm
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()
  cfg.seed = 42

  # Dataset
  cfg.train_ds = _make_ds(training=True)

  # Model
  cfg.model = kd.nn.Vit(
      image="batch.image",
      encoder=kd.nn.VitEncoder.from_variant_str("ViT-Ti/8"),
      num_classes=1000,
  )

  # By default, model params are replicated across all devices, but sharding
  # can be explicitly overwritten
  cfg.sharding = kd.sharding.ShardingStrategy(
      # TODO(epot): Use more complicated sharding.
      params=kd.sharding.REPLICATED,
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
      # Track the standard deviation of the attention weights in layer 11.
      "final_attention_std": kd.metrics.Std(
          # All intermediates values from any modules can be accessed
          # through `interms.`
          values="interms.encoder.layers_11.attention.attn_weights[0]"
      ),
      # Monitor the norm of all the parameter vectors and matrices in the model.
      "param_norm": kd.metrics.TreeMap(
          metric=kd.metrics.Norm(tensor="params", axis=None)
      ),
      # Monitor the norm of the gradients of all parameters (concatenated).
      # SkipIfMissing ensures that the metric does not crash if the gradients
      # are None (e.g. when running in eval mode).
      "grad_norm": kd.metrics.SkipIfMissing(
          kd.metrics.TreeReduce(
              metric=kd.metrics.Norm(
                  tensor="grads", axis=None, aggregation_type="concat"
              )
          )
      ),
  }

  cfg.train_summaries = {
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

  # Use a named chain to construct the optimizer.
  # This improves readability of the config and the sweeps because the path
  # becomes "optimizer.adam.b1" rather than "optimizer[1].b1".
  # This also makes the state of the optimizer (as stored in the checkpoint and
  # in context) a dictionary instead of a tuple.
  cfg.optimizer = kd.optim.named_chain(**{
      "clip": optax.clip_by_global_norm(max_norm=1.0),
      "adam": optax.scale_by_adam(b1=0.95),
      "decay": optax.add_decayed_weights(weight_decay=0.1),
      "lr": optax.scale_by_learning_rate(cfg.ref.schedules["learning_rate"]),
  })

  # Checkpointer
  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=1000,
      max_to_keep=1,
  )

  cfg.evals = {
      "eval": kd.evals.Evaluator(
          # Evals can be run either along train or as separate jobs. Here, we
          # launch eval in a separate job and optionally overwrite the
          # platform used.
          run=kd.evals.StandaloneEveryCheckpoint(
              platform="a100=1",
          ),
          num_batches=10,
          ds=_make_ds(training=False),
      )
  }

  # The train job resources can be hardcoded in the config
  GiB = 1024**3  # pylint:disable=invalid-name
  cfg.xm_job = kxm.Job(
      platform="jf=2x2",
      requirements=xm.JobRequirements(
          ram=384 * GiB,
          tmp_ram_fs=8 * GiB,
      ),
  )

  # Checkify enable additional checks within `jax.jit` (NaN, div by 0,...) but
  # Can affect performances
  cfg.checkify_error_categories = checkify.all_checks

  # By default, Kauldron only profile Python on the main process. All
  # processes can be profiled with `all_host=True`
  cfg.profiler = kd.inspect.Profiler(all_host=True)

  return cfg


def _make_ds(training: bool):
  return kd.data.tf.Tfds(
      name="imagenet_resized/64x64",
      split="train" if training else "validation",
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      transforms=[
          kd.data.Elements(keep=["image", "label"]),
          kd.data.ValueRange(key="image", in_vrange=(0, 255), vrange=(0, 1)),
          kd.data.Rearrange(key="label", pattern="... -> ... 1"),
      ],
      batch_size=512,
  )


# multiple named sweeps can be combined using --xp.sweep=lr,seed
def sweep_lr():
  for lr in [0.001, 0.01]:
    yield {
        "schedules.learning_rate.peak_value": lr,
    }


def sweep_seed():
  for seed in [1, 42, 1337]:
    yield {
        "seed": seed,
    }
