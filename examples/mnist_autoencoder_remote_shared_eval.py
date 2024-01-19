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

```sh
xmanager launch third_party/py/kauldron/xm/launch.py -- \
  --cfg=third_party/py/kauldron/examples/mnist_autoencoder_remote_shared_eval.py \
  --xp.platform=jf=2x2
```

"""

from kauldron import konfig
from examples import mnist_autoencoder

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  import tensorflow_datasets as tfds
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = mnist_autoencoder.get_config()

  cfg.num_train_steps = 10_000  # Longer train step to trigger multiple evals

  shared_run = kd.evals.RunSharedXM(shared_name="evals")

  cfg.evals = {
      "eval_train": kd.evals.Evaluator(
          run=shared_run,
          num_batches=None,
          ds=_make_ds(split="train"),
          metrics={},
      ),
      "eval_test": kd.evals.Evaluator(
          run=shared_run,
          num_batches=None,
          ds=_make_ds(split="test"),
          metrics={},
      ),
  }

  return cfg


def _make_ds(split: str):
  return kd.data.PyGrainPipeline(
      data_source=tfds.data_source("mnist", split=split),
      shuffle=False,
      transformations=[
          kd.data.Elements(keep=["image"]),
          kd.data.ValueRange(key="image", vrange=(0, 1)),
      ],
      batch_size=256,
  )
