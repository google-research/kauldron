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

"""Test."""

import os
import pathlib

from kauldron import kd
from kauldron.examples import mnist_autoencoder


def test_multi(tmp_path: pathlib.Path):
  cfg = mnist_autoencoder.get_config()
  cfg.workdir = os.fspath(tmp_path)
  cfg.eval_ds = cfg.train_ds

  with kd.konfig.mock_modules():
    cfg.evals = {
        'test_eval': kd.evals.Evaluator(
            run_every=1,
            num_batches=1,
        ),
        'eval002': kd.evals.Evaluator(
            run_every=1,
            num_batches=1,
        ),
    }

  cfg = kd.konfig.resolve(cfg)

  assert cfg.evals['test_eval'].name == 'test_eval'
  assert cfg.evals['eval002'].name == 'eval002'
