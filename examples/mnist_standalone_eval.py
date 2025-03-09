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

This example shows how to run evaluation in a separate job. Each work-unit
will contain 3 jobs:

  - train: The main training job
  - grouped_evals: Executing `eval_train` and `eval_test`.
  - isolated_eval: Another standalone eval job, running on a different platform.

/!\ Remote evaluators are not yet supported in open-source!!!

"""

from kauldron import konfig
from examples import mnist_autoencoder

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = mnist_autoencoder.get_config()

  cfg.num_train_steps = 10_000  # Longer train step to trigger multiple evals

  # Shared job for both `eval_train` and `eval_test`.
  shared_run = kd.evals.StandaloneEveryCheckpoint(job_group="grouped_evals")
  cfg.evals = {
      "eval_train": kd.evals.Evaluator(
          run=shared_run,
          num_batches=None,
          ds=_make_ds(split="train"),
          metrics={
              "ssim": kd.metrics.Ssim(
                  pred="preds.image",
                  target="batch.image",
              ),
          },
      ),
      "eval_test": kd.evals.Evaluator(
          run=shared_run,
          num_batches=None,
          ds=_make_ds(split="test"),
          metrics={},
      ),
      # Eval other uses a different `run` without `job_group=`, so is run in
      # a separate job.
      "isolated_eval": kd.evals.Evaluator(
          run=kd.evals.StandaloneEveryCheckpoint(),
          num_batches=None,
          ds=_make_ds(split="test"),
          metrics={
              "ssim": kd.metrics.Ssim(
                  pred="preds.image",
                  target="batch.image",
              ),
          },
      ),
  }

  return cfg


def _make_ds(split: str):
  # return kd.data.py.Tfds(
  return kd.data.Tfds(
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
