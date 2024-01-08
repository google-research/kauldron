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

"""Base config."""

from kauldron import konfig
from kauldron import kontext

with konfig.imports():
  # pylint: disable=g-import-not-at-top
  from kauldron import kd
  from projects.nerf import nerf
  import optax
  # pylint: enable=g-import-not-at-top

BATCH = kontext.path_builder_from('batch', nerf.structs.Batch)
PREDS = kontext.path_builder_from('preds', nerf.structs.RayPreds)


def get_config():
  """Returns the base config."""

  cfg = kd.train.Trainer(
      workdir='/tmp/kd/nerf',
      train_ds=_get_ds(training=True),
      model=nerf.nn.NerfRender(
          ray=BATCH.ray,
          mlp=nerf.nn.MLP(),
      ),
      optimizer=optax.adam(learning_rate=0.003),
      evals={
          'eval_train': _get_eval('train'),
          'eval_test': _get_eval('test'),
      },
  )
  return cfg


def _get_eval(split: str) -> kd.evals.Evaluator:
  return kd.evals.Evaluator(
      run=kd.evals.RunEvery(100),
      ds=_get_ds(training=False, split=split),
      num_batches=3,
      metrics={
          'psnr': kd.metrics.Psnr(pred=PREDS.rgb, target=BATCH.rgb),
          'ssim': kd.metrics.Ssim(pred=PREDS.rgb, target=BATCH.rgb),
          'lpips': kd.metrics.LpipsVgg(pred=PREDS.rgb, target=BATCH.rgb),
      },
  )


def _get_ds(*, training: bool, split: str = 'train') -> nerf.data.Pipeline:
  if training:
    data_source = nerf.data.RaySampler(
        num_samples=128,
    )
    # Num samples is the actual batch_size.
    batch_size = 0
  else:
    data_source = nerf.data.ImageSampler()
    batch_size = 32
  return nerf.data.Pipeline(
      scene=nerf.data.Blender(
          name='lego',
          split=split,
      ),
      data_source=data_source,
      batch_size=batch_size,
      shuffle=True if training else False,
  )
