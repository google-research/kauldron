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

r"""Example for meta configs.

```sh
python -m kauldron.main \
    --cfg=examples/tiny_vit_imagenet.py \
    --cfg.workdir=/tmp/kauldron_oss/workdir
```

"""

from copy import deepcopy  # pylint: disable=g-importing-member
import dataclasses
from kauldron import konfig


with konfig.imports():
  # pylint: disable=g-import-not-at-top
  from kauldron import kd
  import optax


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
  """Config for metaconfig demo."""

  model: str = "tiny_vit"
  model_variant: str = "ViT-Ti/8"
  dataset_name: str = "mnist"
  dataset_train_split: str = "train"
  dataset_eval_splits: tuple[str, ...] = ("val", "test")

  def build(self) -> kd.train.Trainer:
    """Creates and returns the config as a konfig.ConfigDict.

    TODO(geco): other options to discuss:
    - get_config() as before
    - property config
    - property configdict

    Returns:
      A `kd.train.Trainer` config.
    """
    cfg = kd.train.Trainer()
    cfg.seed = 42
    cfg.num_train_steps = 1000

    num_classes = dict(mnist=10, imagenet=1000)[self.dataset_name]

    if self.dataset_name == "mnist":
      cfg.train_ds = kd.data.py.Tfds(
          name="mnist",
          split=self.dataset_train_split,
          shuffle=True,
          num_epochs=None,
          transforms=[
              kd.data.Elements(keep=["image"]),
              kd.data.ValueRange(key="image", vrange=(0, 1)),
          ],
          batch_size=256,
      )
    elif self.dataset_name == "imagenet":
      cfg.train_ds = kd.data.py.Tfds(
          name="imagenet2012",
          split=self.dataset_train_split,
          shuffle=True,
          num_epochs=None,
          transforms=[
              kd.data.Elements(keep=["image"]),
              kd.data.ValueRange(key="image", vrange=(0, 1)),
          ],
          batch_size=256,
      )
    else:
      raise ValueError("Unknown dataset")

    if self.model == "tiny_vit":
      cfg.model = kd.nn.Vit(
          image="batch.image",
          encoder=kd.nn.VitEncoder.from_variant_str(self.model_variant),
          num_classes=num_classes,
      )
    else:
      raise ValueError("Unknown model", self.model)

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
    }

    cfg.optimizer = optax.adam(learning_rate=1e-4)

    cfg.evals = {}
    for eval_split in self.dataset_eval_splits:
      cfg.evals[eval_split] = kd.evals.Evaluator(
          run=kd.evals.EveryNSteps(1000),
          num_batches=10,
          ds=get_test_ds_from_train_ds_config(cfg.train_ds, split=eval_split),
      )

    return cfg


class MyConfig(Config):
  """Config for metaconfig demo with reduced number of train steps."""

  def build(self) -> kd.train.Trainer:
    cfg = super().build()
    cfg.num_train_steps = 100
    return cfg


@konfig.ref_fn
def get_test_ds_from_train_ds_config(
    train_ds_config: konfig.ConfigDict, split: str = "test"
) -> konfig.ConfigDict:
  test_ds_config = deepcopy(train_ds_config)
  test_ds_config.num_epochs = 1
  test_ds_config.shuffle = False
  test_ds_config.batch_drop_remainder = False

  test_ds_config.split = split
  return test_ds_config


def sweep_trainsteps():
  yield {"num_train_steps": 11}
  yield {"num_train_steps": 22}


def sweep_dataset():
  # TODO(geco): discuss other options: __init_args__, __build_args__
  yield {"__class_args__.dataset_name": "mnist"}
  yield {"__class_args__.dataset_name": "imagenet"}
