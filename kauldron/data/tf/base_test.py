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

import dataclasses
import functools
import pathlib
from unittest import mock

import grain.tensorflow as grain
from kauldron import kd
from kauldron.data.tf import base as tf_base
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


class _DummyFilterTransform(grain.FilterTransform):
  """Filter to create unknown cardinality dataset."""

  def filter(self, features):
    return features['id'] < 10


class _UnshapedFirstLeafTransform(grain.MapTransform):
  """Inserts a tensor with unknown static rank as the first dictionary key."""

  def map(self, features):
    unshaped = tf.py_function(
        func=lambda x: x,
        inp=[features['id']],
        Tout=tf.int64,
    )
    return {'00_unshaped': unshaped, **features}


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
  ds = ds_cls(  # pyrefly: ignore[missing-argument]
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=False,
  )
  assert len(ds) == 200
  assert list(ds.element_spec.keys()) == ['id', 'rand']
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
  ds = ds_cls(  # pyrefly: ignore[missing-argument]
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

  ds = ds_cls(  # pyrefly: ignore[missing-argument]
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=True,
  )
  exs2 = list(ds)
  assert len(exs2) == 200
  assert exs2[:100] == exs  # First epoch is the same
  # The second epoch has different shuffling.
  assert _ids(exs2) != ids + ids

  ds = ds_cls(  # pyrefly: ignore[missing-argument]
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
        ds = ds_cls(  # pyrefly: ignore[missing-argument]
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
  ds2 = dummy_tfds_legacy_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=False,
      transforms=[
          _DummyRandomTransform(),
          kd.data.Elements(rename={'id': 'code'}),
          kd.contrib.data.AddConstants({'src': 1}),
      ],
  )
  dsmix = kd.data.tf.SampleFromDatasets([ds1, ds2], seed=0)
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
  if issubclass(ds_cls.func, kd.data.tf.TfdsLegacy):  # pyrefly: ignore[missing-attribute]
    checkpoint_kwargs['checkpoint'] = True

  def _make_ds_iter():
    # Load the dataset.
    ds = ds_cls(
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

  ckpt.save(ds_iter, step=1)

  ds_iter = _make_ds_iter()
  ds_iter = ckpt.restore(ds_iter)
  for ex in exs[45:]:
    assert next(ds_iter) == ex


@with_ds_cls
@pytest.mark.parametrize(
    'batch_drop_remainder',
    [
        True,
        False,
        'drop',
        'keep',
        'pad',
        'pad_with_mask',
        kd.data.tf.DropRemainder.DROP,
        kd.data.tf.DropRemainder.KEEP,
        kd.data.tf.DropRemainder.PAD,
        kd.data.tf.DropRemainder.PAD_WITH_MASK,
    ],
)
def test_drop_remainder(
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
    batch_drop_remainder: bool | str | kd.data.tf.DropRemainder,
):  # pylint: disable=redefined-outer-name
  assert kd.data.DropRemainder is kd.data.tf.DropRemainder

  ds = ds_cls(  # pyrefly: ignore[missing-argument]
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=30,
      batch_drop_remainder=batch_drop_remainder,
  )
  batches = list(ds)

  if batch_drop_remainder in (True, 'drop', kd.data.tf.DropRemainder.DROP):
    assert len(ds) == 3
    assert len(batches) == 3
    assert [batch['id'].shape for batch in batches] == [(30,), (30,), (30,)]
    np.testing.assert_array_equal(batches[0]['id'], np.arange(0, 30))
    np.testing.assert_array_equal(batches[1]['id'], np.arange(30, 60))
    np.testing.assert_array_equal(batches[2]['id'], np.arange(60, 90))
    assert '__is_padding__' not in ds.element_spec
    for batch in batches:
      assert '__is_padding__' not in batch
  elif batch_drop_remainder in (False, 'keep', kd.data.tf.DropRemainder.KEEP):
    assert len(ds) == 4
    assert len(batches) == 4
    assert [batch['id'].shape for batch in batches] == [
        (30,),
        (30,),
        (30,),
        (10,),
    ]
    np.testing.assert_array_equal(batches[0]['id'], np.arange(0, 30))
    np.testing.assert_array_equal(batches[1]['id'], np.arange(30, 60))
    np.testing.assert_array_equal(batches[2]['id'], np.arange(60, 90))
    np.testing.assert_array_equal(batches[3]['id'], np.arange(90, 100))
    assert '__is_padding__' not in ds.element_spec
    for batch in batches:
      assert '__is_padding__' not in batch
  elif batch_drop_remainder in ('pad', kd.data.tf.DropRemainder.PAD):
    assert len(ds) == 4
    assert len(batches) == 4
    assert ds.element_spec['id'].shape == (30,)
    assert [batch['id'].shape for batch in batches] == [
        (30,),
        (30,),
        (30,),
        (30,),
    ]
    np.testing.assert_array_equal(batches[0]['id'], np.arange(0, 30))
    np.testing.assert_array_equal(batches[1]['id'], np.arange(30, 60))
    np.testing.assert_array_equal(batches[2]['id'], np.arange(60, 90))
    expected_last_ids = np.concatenate(
        [np.arange(90, 100), np.zeros(20, dtype=np.int64)]
    )
    np.testing.assert_array_equal(batches[3]['id'], expected_last_ids)
    assert '__is_padding__' not in ds.element_spec
    for batch in batches:
      assert '__is_padding__' not in batch
  elif batch_drop_remainder in (
      'pad_with_mask',
      kd.data.tf.DropRemainder.PAD_WITH_MASK,
  ):
    assert len(ds) == 4
    assert len(batches) == 4
    assert ds.element_spec['id'].shape == (30,)
    assert ds.element_spec['__is_padding__'].shape == (30,)
    assert ds.element_spec['__is_padding__'].dtype == np.bool_
    for batch in batches:
      assert batch['id'].shape == (30,)
      assert batch['__is_padding__'].shape == (30,)
    np.testing.assert_array_equal(batches[0]['id'], np.arange(0, 30))
    np.testing.assert_array_equal(batches[1]['id'], np.arange(30, 60))
    np.testing.assert_array_equal(batches[2]['id'], np.arange(60, 90))
    for batch in batches[:3]:
      assert not np.any(batch['__is_padding__'])
    expected_last_ids = np.concatenate(
        [np.arange(90, 100), np.zeros(20, dtype=np.int64)]
    )
    np.testing.assert_array_equal(batches[3]['id'], expected_last_ids)
    expected_last_mask = np.array([False] * 10 + [True] * 20)
    np.testing.assert_array_equal(
        batches[3]['__is_padding__'], expected_last_mask
    )


@with_ds_cls
def test_invalid_drop_remainder(
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  with pytest.raises(ValueError):
    ds_cls(  # pyrefly: ignore[missing-argument]
        data_dir=dummy_builder.data_dir_root,
        num_epochs=1,
        shuffle=False,
        batch_size=30,
        batch_drop_remainder='keepp',
    )


def test_multi_host_pad_with_mask(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  with mock.patch('jax.process_count', return_value=4):
    for process_index in range(4):
      with (
          mock.patch('jax.process_index', return_value=process_index),
          mock.patch(
              'jax.experimental.multihost_utils.process_allgather',
              return_value=np.array([25, 25, 24, 24]),
          ),
      ):
        ds = dummy_tfds_ds(
            data_dir=dummy_builder.data_dir_root,
            split='train[:98]',
            num_epochs=1,
            shuffle=False,
            batch_size=32,
            batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
        )
        assert len(ds) == 4
        batches = list(ds)
        assert len(batches) == 4
        for batch in batches:
          assert batch['id'].shape == (8,)
          assert batch['__is_padding__'].shape == (8,)
        for batch in batches[:3]:
          assert not np.any(batch['__is_padding__'])

        if process_index in (0, 1):
          assert not batches[3]['__is_padding__'][0]
          assert np.all(batches[3]['__is_padding__'][1:])
        else:
          assert np.all(batches[3]['__is_padding__'])


@pytest.mark.parametrize(
    'batch_drop_remainder',
    [
        kd.data.tf.DropRemainder.PAD,
        kd.data.tf.DropRemainder.PAD_WITH_MASK,
    ],
)
def test_unknown_cardinality_pad_and_pad_with_mask(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
    batch_drop_remainder: kd.data.tf.DropRemainder,
):  # pylint: disable=redefined-outer-name
  ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=3,
      transforms=[_DummyFilterTransform()],
      batch_drop_remainder=batch_drop_remainder,
  )
  assert ds.element_spec['id'].shape == (3,)
  if batch_drop_remainder == kd.data.tf.DropRemainder.PAD_WITH_MASK:
    assert ds.element_spec['__is_padding__'].shape == (3,)

  batches = list(ds)
  assert len(batches) == 4
  for batch in batches:
    assert batch['id'].shape == (3,)

  np.testing.assert_array_equal(batches[0]['id'], [0, 1, 2])
  np.testing.assert_array_equal(batches[1]['id'], [3, 4, 5])
  np.testing.assert_array_equal(batches[2]['id'], [6, 7, 8])
  np.testing.assert_array_equal(batches[3]['id'], [9, 0, 0])

  if batch_drop_remainder == kd.data.tf.DropRemainder.PAD_WITH_MASK:
    for batch in batches:
      assert batch['__is_padding__'].shape == (3,)
    for batch in batches[:3]:
      assert not np.any(batch['__is_padding__'])
    np.testing.assert_array_equal(
        batches[3]['__is_padding__'], [False, True, True]
    )
  else:
    for batch in batches:
      assert '__is_padding__' not in batch


def test_pad_batch_skips_unshaped_first_leaf(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=3,
      transforms=[_DummyFilterTransform(), _UnshapedFirstLeafTransform()],
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
  )
  batches = list(ds)
  assert len(batches) == 4
  np.testing.assert_array_equal(batches[3]['id'], [9, 0, 0])
  np.testing.assert_array_equal(batches[3]['00_unshaped'], [9, 0, 0])
  np.testing.assert_array_equal(
      batches[3]['__is_padding__'], [False, True, True]
  )


def test_first_shaped_tensor_raises_on_no_valid_leaf():
  with pytest.raises(
      ValueError, match='Could not find any tensor leaf with rank >= 1'
  ):
    tf_base._first_shaped_tensor({'scalar': tf.constant(1), 'empty': None})
