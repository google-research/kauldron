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

from __future__ import annotations

from etils import epath
from flax import linen as nn
import jax
import jax.numpy as jnp
from kauldron import kd
import numpy as np


class ModelWithBatchNorm(nn.Module):
  image: kd.kontext.Key = 'batch.image'

  @nn.compact
  def __call__(self, image: jax.Array) -> jax.Array:
    x = nn.Dense(features=4)(image)
    x = nn.BatchNorm(use_running_average=False)(x)
    return x


def test_export_with_collections(tmp_path: epath.Path):
  exporter = kd.export.JaxModelExporter(
      workdir=tmp_path,
      rng_streams=kd.train.RngStreams(),
      ds_sharding=kd.sharding.FIRST_DIM,
  )
  model = ModelWithBatchNorm()
  init_vars = model.init(jax.random.PRNGKey(0), jnp.zeros((2, 8)))
  params = init_vars['params']
  collections = {k: v for k, v in init_vars.items() if k != 'params'}
  assert 'batch_stats' in collections

  state = kd.train.TrainState(
      step=0,
      params=params,
      opt_state={},
      collections=collections,
  )
  elem_spec = {'image': np.zeros((2, 8), dtype=np.float32)}

  exporter.export(
      model=model, state=state, element_spec=elem_spec, is_training=False
  )

  exported_path = tmp_path / 'train_model.jax_exported'
  assert exported_path.exists()

  # Load the exported model and verify it can be evaluated.
  exported = jax.export.deserialize(bytearray(exported_path.read_bytes()))
  inputs = jnp.ones((2, 8))
  out = exported.call(
      params=params,
      key=jax.random.PRNGKey(0),
      image=inputs,
      collections=collections,
  )
  assert out['preds'].shape == (2, 4)
  assert 'interms' in out

  # Verify numerical equivalence with standard model.apply.
  expected_preds, _ = model.apply(
      {'params': params} | collections,
      image=inputs,
      mutable=True,
  )
  np.testing.assert_allclose(out['preds'], expected_preds, atol=1e-5)
