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

import dataclasses

from kauldron import kd
from kauldron.train import metric_writer


def test_trainer_replace():
  trainer = kd.train.Trainer(
      eval_ds=kd.data.py.Tfds(name='mnist', split='train', shuffle=False),
      train_ds=kd.data.py.Tfds(
          name='mnist', split='train', shuffle=True, seed=60
      ),
      init_transform=kd.ckpts.PartialKauldronLoader(workdir='/some/workdir'),
      evals={
          'eval': kd.evals.Evaluator(
              run=kd.evals.EveryNSteps(100),
          ),
      },
      seed=0,
      model=None,
      optimizer=None,
  )

  assert trainer.eval_ds.seed == 0
  assert trainer.train_ds.seed == 60
  assert trainer.evals['eval'].ds.seed == 0  # pytype: disable=attribute-error
  assert isinstance(trainer.evals['eval'].writer, metric_writer.KDMetricWriter)
  assert trainer.trainstep.init_transform.workdir == '/some/workdir'  # pytype: disable=attribute-error

  # Replacing the trainer values are correctly propagated.
  new_trainer = dataclasses.replace(
      trainer,
      seed=42,
      init_transform=kd.ckpts.PartialKauldronLoader(workdir='/new/workdir'),
      writer=metric_writer.NoopWriter(),
  )

  assert new_trainer.eval_ds.seed == 42
  assert new_trainer.train_ds.seed == 60
  assert new_trainer.evals['eval'].ds.seed == 42  # pytype: disable=attribute-error
  assert isinstance(new_trainer.evals['eval'].writer, metric_writer.NoopWriter)
  assert new_trainer.trainstep.init_transform.workdir == '/new/workdir'  # pytype: disable=attribute-error
