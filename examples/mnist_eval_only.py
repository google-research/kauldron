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

r"""Example of an eval-only job.

This require to first run the `mnist_autoencoder.py` training. Then this
config will perform a separate evaluation.

```sh
python -m kauldron.main \
    --cfg=examples/mnist_eval_only.py \
    --cfg.workdir=/tmp/kauldron_oss/workdir
```

"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
# pylint: enable=g-import-not-at-top


def get_config():
  """Eval-only config."""
  cfg = kd.train.Trainer.eval_only()

  cfg.evals = {}
  for split in ["train", "test"]:
    cfg.evals[f"eval_{split}"] = kd.evals.Evaluator(
        run=kd.evals.StandaloneLastCheckpoint(),
        num_batches=None,
        ds=_make_ds(split=split),
        losses={
            "recon": kd.losses.L2(preds="preds.image", targets="batch.image"),
        },
        summaries={
            "gt": kd.summaries.ShowImages(images="batch.image", num_images=5),
            "recon": kd.summaries.ShowImages(
                images="preds.image", num_images=5
            ),
        },
    )

  return cfg


def _make_ds(split: str):
  return kd.data.tf.Tfds(
      name="mnist",
      split=split,
      shuffle=False,
      num_epochs=1,
      transforms=[
          kd.data.Elements(keep=["image"]),
          kd.data.ValueRange(key="image", vrange=(0, 1)),
      ],
      batch_size=256,
  )
