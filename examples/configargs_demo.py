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

r"""Example for meta configs.

```sh
python -m kauldron.main \
    --cfg=examples/configargs_demo.py \
    --cfg.__args__.dataset_name=imagenet \
    --cfg.workdir=/tmp/kauldron_oss/workdir
```

"""

import dataclasses
from kauldron import konfig


with konfig.imports():
  # pylint: disable=g-import-not-at-top
  from kauldron import kd
  import optax
  from flax import linen as nn


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigArgs:
  """Config for metaconfig demo."""

  model: str = "tiny_vit"
  model_variant: str = "ViT-Ti/8"
  dataset_names: tuple[str, ...] = ("imagenet2012", "coco", "cifar100")
  dataset_weights: tuple[float, ...] = (1.0, 0.5, 1.0)
  dataset_train_split: str = "train"
  dataset_eval_splits: tuple[str, ...] = ("test",)
  batch_size: int = 256


def get_config(args: ConfigArgs = ConfigArgs()):
  """Creates and returns the config as a konfig.ConfigDict.

  Args:
    args: ConfigArgs dataclass.

  Returns:
    A `kd.train.Trainer` config.
  """
  cfg = kd.train.Trainer()
  cfg.seed = 42
  cfg.num_train_steps = 1000

  # save args in aux
  cfg.aux = args

  # Model
  if args.model == "flat_autoencoder":
    cfg.model = kd.nn.FlatAutoencoder(
        inputs="batch.image",
        encoder=nn.Dense(features=128),
        decoder=nn.Dense(features=32 * 32 * 3),
    )

  elif args.model == "tiny_vit":
    # TODO(geco): add module that returns image key
    cfg.model = kd.nn.VitAutoEncoder(
        encoder=kd.nn.VitEncoder.from_variant_str(args.model_variant),
        image="batch.image",
    )

  else:
    raise ValueError("Unknown model", args.model)

  # Dataset
  cfg.train_ds = _make_ds(args, split=args.dataset_train_split)

  # Losses
  cfg.train_losses = {
      "recon": kd.losses.L2(preds="preds.image", targets="batch.image"),
  }

  # Metrics
  cfg.train_metrics = {
      "latent_norm": kd.metrics.Norm(tensor="interms.encoder.__call__[0]"),
  }

  cfg.train_summaries = {
      "gt": kd.summaries.ShowImages(images="batch.image", num_images=5),
      "recon": kd.summaries.ShowImages(images="preds.image", num_images=5),
  }

  cfg.optimizer = optax.adam(learning_rate=1e-4)

  cfg.evals = {}
  for eval_split in args.dataset_eval_splits:
    cfg.evals[eval_split] = kd.evals.Evaluator(
        run=kd.evals.EveryNSteps(1000),
        num_batches=10,
        ds=_make_ds(args, split=eval_split, training=False),
    )

  return cfg


def _make_ds(args: ConfigArgs, split: str = "train", training: bool = True):
  """Creates and returns the dataset as a konfig.ConfigDict.

  Because args.dataset_names is from args,
  it can be iterated over, contrarily to a reference.

  Args:
    args: ConfigArgs dataclass.
    split: The dataset split to use.
    training: Whether the dataset is used for training.

  Returns:
    A `konfig.ConfigDict` configuring the dataset.
  """
  datasets = [
      kd.data.py.Tfds(
          name=dataset_name,
          split=split,
          shuffle=training,
          num_epochs=None if training else 1,
          transforms=[
              kd.data.Elements(keep=["image"]),
              kd.data.Resize(key="image", size=(64, 64)),
              kd.data.ValueRange(key="image", vrange=(-1, 1)),
          ],
      )
      for dataset_name in args.dataset_names
  ]
  return kd.data.py.Mix(
      datasets=datasets,
      weights=list(args.dataset_weights),
      seed=0,
      batch_size=args.batch_size,
      transforms=[kd.data.py.RandomCrop(key="image", shape=(32, 32, None))],
  )


def sweep_main():
  for lr in (1e-4, 1e-5):
    for dataset_names, dataset_weights in (
        (("imagenet2012",), (1.0,)),
        (("imagenet2012", "coco"), (1.0, 1.0)),
        (("imagenet2012", "coco", "cifar100"), (1.0, 1.0, 1.0)),
    ):
      yield {
          "optimizer.learning_rate": lr,
          "__args__.dataset_name": dataset_names,
          "__args__.dataset_weights": dataset_weights,
      }
