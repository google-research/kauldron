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

"""Tests."""

import dataclasses
import functools
import pathlib
from unittest import mock

import grain.tensorflow as grain
from kauldron import kd
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds


class _DummyRandomTransform(grain.RandomMapTransform):
  """Test random transformation determinism."""

  def random_map(self, features, seed):
    features['rand'] = tf.cast(
        features['id'], tf.float32
    ) + tf.random.stateless_uniform([], seed)
    return features


dummy_tfds_ds = functools.partial(
    kd.data.tf.Tfds,
    name='dummy_dataset',
    split='train',
    seed=0,
    transforms=[_DummyRandomTransform()],
)


dummy_tfds_legacy_ds = functools.partial(
    kd.data.tf.TfdsLegacy,
    name='dummy_dataset',
    split='train',
    seed=0,
    shuffle_buffer_size=100,
    transforms=[_DummyRandomTransform()],
)

_TfdsCls = type[kd.data.tf.Tfds | kd.data.tf.TfdsLegacy]


with_ds_cls = pytest.mark.parametrize(
    'ds_cls', [dummy_tfds_ds, dummy_tfds_legacy_ds]  # TODO(epot): Add SeqIO
)


class DummyDataset(tfds.core.GeneratorBasedBuilder):
  """Minimal DatasetBuilder."""

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({'id': np.int64}),
        description='Minimal DatasetBuilder.',
        disable_shuffling=True,
    )

  def _split_generators(self, dl_manager):
    del dl_manager
    return {
        'train': self._generate_examples(),
    }

  def _generate_examples(self):
    for i in range(100):
      yield i, {'id': i}


def _ids(ds, key: str = 'id') -> list[int]:
  return [ex[key] for ex in ds]


def _rngs(ds):
  return _ids(ds, key='rand')


@pytest.fixture(scope='module')
def dummy_builder(tmp_path_factory: pytest.TempPathFactory):
  tmp_path = tmp_path_factory.mktemp('data_dir')
  builder = DummyDataset(data_dir=tmp_path)
  builder.download_and_prepare(file_format='array_record')
  yield builder


@with_ds_cls
def test_no_shuffle(
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds = ds_cls(  # pytype: disable=wrong-keyword-args,missing-parameter
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=False,
  )
  assert len(ds) == 200
  assert list(ds.element_spec.keys()) == ['id', 'rand']  # pytype: disable=attribute-error
  exs = list(ds)
  assert _ids(exs) == list(range(100)) + list(range(100))
  # Rng across epochs are different.
  assert _rngs(exs)[:100] != _rngs(exs)[100:]

  # Iterating twice on the dataset should be deterministic.
  assert list(ds) == exs


@with_ds_cls
def test_shuffle(
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds = ds_cls(  # pytype: disable=wrong-keyword-args,missing-parameter
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=True,
  )
  assert len(ds) == 100

  # Iterating twice on the dataset yields the same results.
  exs = list(ds)
  ids = _ids(exs)
  if isinstance(ds, kd.data.tf.WithShuffleBuffer):
    # When `ds.shuffle` is used, the dataset has to be re-created as
    # `reshuffle_each_iteration` is persistent on the cached iterator.
    ds = dataclasses.replace(ds)
  assert list(ds) == exs

  ds = ds_cls(  # pytype: disable=wrong-keyword-args,missing-parameter
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=True,
  )
  exs2 = list(ds)
  assert len(exs2) == 200
  assert exs2[:100] == exs  # First epoch is the same
  # The second epoch has different shuffling.
  assert _ids(exs2) != ids + ids

  ds = ds_cls(  # pytype: disable=wrong-keyword-args,missing-parameter
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=True,
      seed=1,  # Different seed
  )
  exs_new_seed = list(ds)
  assert len(exs_new_seed) == len(ids)
  assert _ids(exs_new_seed) != ids  # Seed change the shuffling
  assert _rngs(exs_new_seed) != _rngs(exs)


@with_ds_cls
def test_sharding(
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  all_ids = set()
  with mock.patch('jax.process_count', return_value=4):
    for process_index in range(4):
      with mock.patch('jax.process_index', return_value=process_index):
        ds = ds_cls(  # pytype: disable=wrong-keyword-args,missing-parameter
            data_dir=dummy_builder.data_dir_root,
            num_epochs=1,
            shuffle=True,
            seed=0,
        )
        assert len(ds) == 25
        ids = _ids(ds)
        assert len(ids) == 25
        all_ids |= set(ids)

  assert len(all_ids) == 100  # No overlapping across shards


def test_sample_from_datasets(dummy_builder: tfds.core.GeneratorBasedBuilder):  # pylint: disable=redefined-outer-name
  """Tests SampleFromDatasets."""
  ds1 = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      transforms=[
          _DummyRandomTransform(),
          kd.data.Elements(rename={'id': 'code'}),
          kd.contrib.data.AddConstants({'src': 0}),
      ],
  )
  ds2 = dummy_tfds_legacy_ds(  # pytype: disable=wrong-keyword-args
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=False,
      transforms=[
          _DummyRandomTransform(),
          kd.data.Elements(rename={'id': 'code'}),
          kd.contrib.data.AddConstants({'src': 1}),
      ],
  )
  dsmix = kd.data.tf.SampleFromDatasets(  # pytype: disable=wrong-keyword-args
      [ds1, ds2], seed=0
  )
  exs = list(dsmix)
  ids = _ids(exs, key='code')
  assert len(ids) == 300
  assert sorted(ids) == sorted(list(range(100)) * 3)

  exs0 = [ex for ex in exs if ex['src'] == 0]
  exs1 = [ex for ex in exs if ex['src'] == 1]
  assert len(exs0) == 100
  assert len(exs1) == 200
  # Not sorted so examples are the same
  assert _ids(exs0, key='code') == _ids(exs1, key='code')[:100]
  # But randomness different across datasets.
  assert _rngs(exs0) != _rngs(exs1)[:100]


@with_ds_cls
def test_checkpoint(
    tmp_path: pathlib.Path,
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ckpt = kd.ckpts.Checkpointer(
      workdir=tmp_path,
      save_interval_steps=1,
  )

  checkpoint_kwargs = {}
  if issubclass(ds_cls.func, kd.data.tf.TfdsLegacy):  # pytype: disable=attribute-error
    checkpoint_kwargs['checkpoint'] = True

  def _make_ds_iter():
    # Load the dataset.
    ds = ds_cls(  # pytype: disable=wrong-keyword-args,missing-parameter
        data_dir=dummy_builder.data_dir_root,
        num_epochs=2,
        shuffle=True,
        seed=0,
        **checkpoint_kwargs,
    )
    return iter(ds)

  exs = list(_make_ds_iter())

  ds_iter = _make_ds_iter()
  for ex in exs[:45]:
    assert next(ds_iter) == ex

  ckpt.save(ds_iter, step=1)  # pytype: disable=wrong-arg-types

  ds_iter = _make_ds_iter()
  ds_iter = ckpt.restore(ds_iter)  # pytype: disable=wrong-arg-types
  for ex in exs[45:]:
    assert next(ds_iter) == ex
