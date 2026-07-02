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

"""Test."""

import typing
from etils import enp
import flax.linen as nn
import jax
from kauldron import inspect
from kauldron.utils.sharding_utils import sharding
import numpy as np


def test_batch():

  batch = {
      'a': np.asarray(['a', 'bb']),
      'b': np.asarray([1, 2, 3]),
      'c': {
          'f': np.asarray([True, False]),
          'g': np.asarray(1),
      },
  }

  df = inspect.get_batch_stats(batch)
  assert len(df) == 4
  assert list(df['Name']) == ['batch.a', 'batch.b', 'batch.c.f', 'batch.c.g']


def test_model_overview():

  class SimpleModel(nn.Module):

    @nn.compact
    def __call__(self, batch):
      x = batch['image']
      return nn.Dense(features=2)(x)

  model = SimpleModel()

  class MockPipeline:

    @property
    def element_spec(self):
      return {
          'image': enp.ArraySpec(shape=(1, 4), dtype=np.float32),
      }

  ds = MockPipeline()
  rngs = {'params': jax.random.key(0)}

  df = inspect.get_colab_model_overview(
      model=model,
      train_ds=typing.cast(typing.Any, ds),
      ds_sharding=sharding.FIRST_DIM,
      rngs=rngs,
  )

  assert df is not None
  # StyledDataFrame wraps pd.DataFrame, so columns should be accessible
  assert 'Path' in df.columns
  assert 'Own Params' in df.columns

  html = df._repr_html_()
  assert 'max-width' in html  # pyrefly: ignore[not-iterable]
  assert '400px' in html  # pyrefly: ignore[not-iterable]
  assert 'word-wrap' in html  # pyrefly: ignore[not-iterable]
  assert 'break-word' in html  # pyrefly: ignore[not-iterable]
