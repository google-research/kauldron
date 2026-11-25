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

"""Tests for Kauldron tf.data pipeline base class and loaders."""

import dataclasses
import functools
import pathlib
from typing import ClassVar
from unittest import mock

from clu import preprocess_spec
from etils import enp
import grain.tensorflow as grain
import jax
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


def test_sample_from_datasets_batched(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds1 = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
  )
  ds2 = dummy_tfds_legacy_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
  )
  dsmix = kd.data.tf.SampleFromDatasets(
      [ds1, ds2],
      seed=0,
      batch_size=30,
      batch_drop_remainder=True,
  )
  assert dsmix.batch_drop_remainder == kd.data.tf.DropRemainder.DROP
  batches = list(dsmix)
  assert len(batches) == 6
  assert [batch['id'].shape for batch in batches] == [(30,)] * 6

  dszip = kd.data.tf.ZipDatasets(
      {'a': ds1, 'b': ds2},
      seed=0,
      batch_size=30,
      batch_drop_remainder=False,
  )
  assert dszip.batch_drop_remainder == kd.data.tf.DropRemainder.KEEP
  zip_batches = list(dszip)
  assert len(zip_batches) == 4
  assert [batch['a']['id'].shape for batch in zip_batches] == [
      (30,),
      (30,),
      (30,),
      (10,),
  ]

  # Single-host `PAD_WITH_MASK` dynamically batches and pads
  # `SampleFromDatasets` even though TF gives `sample_from_datasets` unknown
  # cardinality (-2).
  ds_child_single = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:15]',
      num_epochs=1,
      shuffle=False,
  )
  dsmix_pad_single = kd.data.tf.SampleFromDatasets(
      [ds_child_single, ds_child_single],
      seed=0,
      batch_size=8,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
  )
  mix_pad_single_batches = list(dsmix_pad_single)
  assert len(mix_pad_single_batches) == 4
  assert mix_pad_single_batches[-1]['id'].shape == (8,)
  assert np.any(mix_pad_single_batches[-1]['__is_padding__'])

  with mock.patch('jax.process_count', return_value=4):
    with (
        mock.patch('jax.process_index', return_value=2),
        mock.patch(
            'jax.experimental.multihost_utils.process_allgather',
            return_value=np.array([25, 25, 24, 24]),
        ),
    ):
      ds_child = dummy_tfds_ds(
          data_dir=dummy_builder.data_dir_root,
          split='train[:98]',
          num_epochs=1,
          shuffle=False,
      )
      # Multi-host `PAD_WITH_MASK` requires known dataset cardinality.
      # `SampleFromDatasets` produces UNKNOWN_CARDINALITY (-2), so it raises.
      dsmix_pad_multi = kd.data.tf.SampleFromDatasets(
          [ds_child, ds_child],
          seed=0,
          batch_size=32,
          batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      )
      with pytest.raises(
          ValueError,
          match='DropRemainder.PAD_WITH_MASK across multiple JAX processes',
      ):
        _ = list(dsmix_pad_multi)

      # `ZipDatasets` preserves static cardinality (min of children), so it
      # synchronizes lengths and pads to equal batch counts across hosts.
      dszip_pad = kd.data.tf.ZipDatasets(
          {'a': ds_child, 'b': ds_child},
          seed=0,
          batch_size=32,
          batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      )
      zip_pad_batches = list(dszip_pad)
      assert len(zip_pad_batches) == 4
      assert not np.any(zip_pad_batches[2]['__is_padding__'])
      assert np.all(zip_pad_batches[3]['__is_padding__'])


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


@pytest.mark.parametrize('known_cardinality', [True, False])
def test_pad_batch_skips_unshaped_first_leaf(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
    known_cardinality: bool,
):  # pylint: disable=redefined-outer-name
  if known_cardinality:
    transforms = [_UnshapedFirstLeafTransform()]
  else:
    transforms = [_DummyFilterTransform(), _UnshapedFirstLeafTransform()]
  ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:10]' if known_cardinality else 'train',
      num_epochs=1,
      shuffle=False,
      batch_size=3,
      transforms=transforms,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
  )
  batches = list(ds)
  assert len(batches) == 4
  np.testing.assert_array_equal(batches[3]['id'], [9, 0, 0])
  np.testing.assert_array_equal(batches[3]['00_unshaped'], [9, 0, 0])
  np.testing.assert_array_equal(
      batches[3]['__is_padding__'], [False, True, True]
  )


@pytest.mark.parametrize(
    'batch_drop_remainder',
    [
        kd.data.tf.DropRemainder.KEEP,
        kd.data.tf.DropRemainder.PAD,
    ],
)
def test_multi_host_keep_and_pad(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
    batch_drop_remainder: kd.data.tf.DropRemainder,
):  # pylint: disable=redefined-outer-name
  # 100 examples across 3 hosts with host_batch_size=33 (batch_size=99):
  # shard_drop_remainder=True gives 33 examples (1 batch of 33) on every host.
  with mock.patch('jax.process_count', return_value=3):
    for process_index in range(3):
      with mock.patch('jax.process_index', return_value=process_index):
        ds = dummy_tfds_ds(
            data_dir=dummy_builder.data_dir_root,
            num_epochs=1,
            shuffle=False,
            batch_size=99,
            batch_drop_remainder=batch_drop_remainder,
        )
        batches = list(ds)
        assert len(batches) == 1
        assert batches[0]['id'].shape == (33,)


def test_multi_host_pad_with_mask_unknown_cardinality_raises(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  with (
      mock.patch('jax.process_count', return_value=2),
      mock.patch('jax.process_index', return_value=0),
  ):
    ds = dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        num_epochs=1,
        shuffle=False,
        batch_size=6,
        transforms=[_DummyFilterTransform()],
        batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
    )
    with pytest.raises(
        ValueError,
        match='DropRemainder.PAD_WITH_MASK across multiple JAX processes',
    ):
      list(ds)


def test_first_shaped_tensor_raises_on_no_valid_leaf():
  with pytest.raises(
      ValueError, match='Could not find any tensor leaf with rank >= 1'
  ):
    tf_base._first_shaped_tensor({'scalar': tf.constant(1), 'empty': None})


class _AddScalarPlusOne(kd.data.MapTransform):
  """Pre-batch transform expecting unbatched scalar `id`."""

  def map(self, element):
    if element['id'].shape.rank != 0:
      raise ValueError(f"Expected scalar `id`, got {element['id'].shape}")
    return element | {'pre': element['id'] + 1}


class _AddBatchSum(kd.data.MapTransform):
  """Post-batch transform computing the sum across the batch dimension."""

  def map(self, element):
    if element['pre'].shape.rank != 1:
      raise ValueError(
          f"Expected rank-1 batched `pre`, got {element['pre'].shape}"
      )
    batch_sum = tf.reduce_sum(element['pre'])
    return element | {
        'batch_sum': tf.broadcast_to(batch_sum, tf.shape(element['pre']))
    }


@dataclasses.dataclass(frozen=True)
class _BatchRandomTransform(grain.RandomMapTransform):
  """Post-batch random transform adding a per-batch uniform offset."""

  out_key: str = 'post_rand'

  def random_map(self, features, seed):
    offset = tf.random.stateless_uniform([], seed)
    features[self.out_key] = tf.cast(features['id'], tf.float32) + offset
    return features


@dataclasses.dataclass(frozen=True)
class _PaddedBatchTransform(grain.UnsafeTfDataTransform):
  """Custom batch transform using `tf.data.Dataset.padded_batch`."""

  batch_size: int

  def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.padded_batch(self.batch_size, drop_remainder=True)


@with_ds_cls
def test_post_batch_transforms(
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds = ds_cls(  # pyrefly: ignore[missing-argument]
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=10,
      transforms=[_AddScalarPlusOne()],
      post_batch_transforms=[
          _AddBatchSum(),
          kd.data.Cast(key='pre', dtype=tf.float32),
      ],
  )
  assert len(ds) == 10
  assert set(ds.element_spec.keys()) == {'id', 'pre', 'batch_sum'}
  assert ds.element_spec['id'].shape == (10,)
  assert ds.element_spec['pre'].shape == (10,)
  assert ds.element_spec['pre'].dtype == np.float32
  assert ds.element_spec['batch_sum'].shape == (10,)
  assert ds.element_spec['batch_sum'].dtype == np.int64

  batches = list(ds)
  assert len(batches) == 10
  np.testing.assert_array_equal(batches[0]['id'], np.arange(10))
  np.testing.assert_array_equal(
      batches[0]['pre'], np.arange(1, 11, dtype=np.float32)
  )
  np.testing.assert_array_equal(
      batches[0]['batch_sum'], np.full((10,), 55, dtype=np.int64)
  )
  np.testing.assert_array_equal(
      batches[-1]['batch_sum'], np.full((10,), 955, dtype=np.int64)
  )


@with_ds_cls
def test_post_batch_transforms_drop_remainder_false(
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds = ds_cls(  # pyrefly: ignore[missing-argument]
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=30,
      batch_drop_remainder=False,
      transforms=[_AddScalarPlusOne()],
      post_batch_transforms=[
          _AddBatchSum(),
          _BatchRandomTransform(),
      ],
  )
  assert len(ds) == 4
  assert ds.element_spec['id'].shape == (None,)
  assert ds.element_spec['batch_sum'].shape == (None,)
  assert ds.element_spec['post_rand'].shape == (None,)

  batches = list(ds)
  assert [b['id'].shape for b in batches] == [(30,), (30,), (30,), (10,)]
  # Final partial batch holds ids 90..99 -> pre 91..100 -> sum = 955.
  np.testing.assert_array_equal(
      batches[-1]['batch_sum'], np.full((10,), 955, dtype=np.int64)
  )


def test_post_batch_transforms_mapping_and_konfig(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds_dict = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=10,
      transforms={'add_one': _AddScalarPlusOne()},
      post_batch_transforms={
          'sum': _AddBatchSum(),
          'scale': lambda ex: ex | {'scaled': ex['batch_sum'] * 2},
      },
  )
  first_batch = next(iter(ds_dict))
  np.testing.assert_array_equal(
      first_batch['scaled'], np.full((10,), 110, dtype=np.int64)
  )

  with kd.konfig.mock_modules():
    cfg = kd.data.tf.Tfds(
        name='dummy_dataset',
        split='train',
        data_dir=dummy_builder.data_dir_root,
        num_epochs=1,
        shuffle=False,
        seed=0,
        batch_size=10,
        transforms={'keep': kd.data.Elements(keep=['id'])},
        post_batch_transforms={
            'cast': kd.data.Cast(key='id', dtype=tf.float32),
            'rename': kd.data.Elements(rename={'id': 'float_id'}),
        },
    )
  ds_resolved = kd.konfig.resolve(cfg)
  resolved_batch = next(iter(ds_resolved))
  assert list(resolved_batch.keys()) == ['float_id']
  np.testing.assert_array_equal(
      resolved_batch['float_id'], np.arange(10, dtype=np.float32)
  )


@with_ds_cls
def test_post_batch_random_transform_and_checkpoint(
    tmp_path: pathlib.Path,
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  checkpoint_kwargs = {}
  if issubclass(ds_cls.func, kd.data.tf.TfdsLegacy):  # pyrefly: ignore[missing-attribute]
    checkpoint_kwargs['checkpoint'] = True

  def _make_ds():
    return ds_cls(  # pyrefly: ignore[missing-argument]
        data_dir=dummy_builder.data_dir_root,
        num_epochs=2,
        shuffle=True,
        seed=0,
        batch_size=10,
        transforms=[_DummyRandomTransform()],
        post_batch_transforms=[
            _BatchRandomTransform(out_key='post_rand_1'),
            _BatchRandomTransform(out_key='post_rand_2'),
        ],
        **checkpoint_kwargs,
    )

  batches1 = list(_make_ds())
  batches2 = list(_make_ds())
  assert len(batches1) == 20
  for b1, b2 in zip(batches1, batches2, strict=True):
    np.testing.assert_array_equal(b1['id'], b2['id'])
    np.testing.assert_array_equal(b1['rand'], b2['rand'])
    np.testing.assert_array_equal(b1['post_rand_1'], b2['post_rand_1'])
    np.testing.assert_array_equal(b1['post_rand_2'], b2['post_rand_2'])

  # Consecutive post-batch RandomMapTransforms receive independent split seeds,
  # and differ from pre-batch elementwise seeds.
  assert not np.array_equal(
      batches1[0]['post_rand_1'], batches1[0]['post_rand_2']
  )
  assert not np.array_equal(batches1[0]['post_rand_1'], batches1[0]['rand'])

  # Per-batch random offset differs across consecutive batches.
  offset0 = batches1[0]['post_rand_1'][0] - batches1[0]['id'][0]
  offset1 = batches1[1]['post_rand_1'][0] - batches1[1]['id'][0]
  assert offset0 != offset1

  ckpt = kd.ckpts.Checkpointer(workdir=tmp_path, save_interval_steps=1)
  ds_iter = iter(_make_ds())
  for expected in batches1[:7]:
    actual = next(ds_iter)
    np.testing.assert_array_equal(
        actual['post_rand_1'], expected['post_rand_1']
    )
    np.testing.assert_array_equal(
        actual['post_rand_2'], expected['post_rand_2']
    )
  ckpt.save(ds_iter, step=1)

  restored_iter = ckpt.restore(iter(_make_ds()))
  for expected in batches1[7:]:
    actual = next(restored_iter)
    np.testing.assert_array_equal(
        actual['post_rand_1'], expected['post_rand_1']
    )
    np.testing.assert_array_equal(
        actual['post_rand_2'], expected['post_rand_2']
    )


@with_ds_cls
def test_post_batch_transforms_sharding(
    ds_cls: _TfdsCls,
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  all_ids = set()
  with mock.patch.object(kd.data.utils.jax, 'process_count', return_value=4):
    for process_index in range(4):
      with mock.patch.object(
          kd.data.utils.jax, 'process_index', return_value=process_index
      ):
        ds = ds_cls(  # pyrefly: ignore[missing-argument]
            data_dir=dummy_builder.data_dir_root,
            num_epochs=1,
            shuffle=True,
            seed=0,
            batch_size=20,  # Global batch size 20 -> host_batch_size 5.
            transforms=[_AddScalarPlusOne()],
            post_batch_transforms=[_AddBatchSum()],
        )
        assert len(ds) == 5
        batches = list(ds)
        assert len(batches) == 5
        for batch in batches:
          assert batch['batch_sum'].shape == (5,)
          np.testing.assert_array_equal(batch['pre'], batch['id'] + 1)
          np.testing.assert_array_equal(
              batch['batch_sum'],
              np.full((5,), batch['pre'].sum(), dtype=np.int64),
          )
          all_ids.update(batch['id'].tolist())

  assert len(all_ids) == 100


def test_post_batch_transforms_sample_from_datasets(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds1 = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      transforms=[_AddScalarPlusOne()],
  )
  ds2 = dummy_tfds_legacy_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      transforms=[
          _AddScalarPlusOne(),
          lambda ex: ex | {'pre': ex['pre'] + 100},
      ],
  )
  dsmix = kd.data.tf.SampleFromDatasets(
      [ds1, ds2],
      weights=[1.0, 0.0],
      stop_on_empty_dataset=True,
      seed=0,
      batch_size=10,
      transforms=[kd.data.Elements(keep=['id', 'pre'])],
      post_batch_transforms=[_AddBatchSum()],
  )
  batches = list(dsmix)
  assert len(batches) == 10
  np.testing.assert_array_equal(
      batches[0]['batch_sum'], np.full((10,), 55, dtype=np.int64)
  )

  # Pre-batched child datasets with unbatched root SampleFromDatasets also work.
  prebatched_mix = kd.data.tf.SampleFromDatasets(
      [
          dataclasses.replace(ds1, batch_size=10),
          dataclasses.replace(ds1, batch_size=10),
      ],
      seed=0,
      post_batch_transforms=[_AddBatchSum()],
  )
  first_prebatched = next(iter(prebatched_mix))
  assert first_prebatched['batch_sum'].shape == (10,)
  np.testing.assert_array_equal(
      first_prebatched['batch_sum'], np.full((10,), 55, dtype=np.int64)
  )

  # Setting post_batch_transforms on an unbatched child dataset raises.
  invalid_sub_ds = dataclasses.replace(
      ds1, post_batch_transforms=[_AddBatchSum()]
  )
  invalid_mix = kd.data.tf.SampleFromDatasets(
      [invalid_sub_ds, ds2],
      seed=0,
      batch_size=10,
  )
  with pytest.raises(ValueError, match='post_batch_transforms'):
    _ = iter(invalid_mix)


def test_post_batch_transforms_zip_datasets(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds_a = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      transforms=[_AddScalarPlusOne()],
  )
  ds_b = dummy_tfds_legacy_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      transforms=[_AddScalarPlusOne()],
  )
  zipped = kd.data.tf.ZipDatasets(
      datasets={'a': ds_a, 'b': ds_b},
      seed=0,
      batch_size=10,
      post_batch_transforms=[
          lambda ex: {'pair_sum': ex['a']['pre'] + ex['b']['pre']}
      ],
  )
  first_batch = next(iter(zipped))
  np.testing.assert_array_equal(
      first_batch['pair_sum'], np.arange(2, 22, 2, dtype=np.int64)
  )


@pytest.mark.parametrize(
    ('batch_size', 'extra_transforms'),
    [
        (None, []),
        (0, []),
        (None, [grain.TfBatchNone()]),
    ],
)
def test_post_batch_transforms_raises_when_unbatched(
    batch_size: int | None,
    extra_transforms: list[grain.Transformation],
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  unbatched_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=batch_size,
      transforms=[_AddScalarPlusOne(), *extra_transforms],
      post_batch_transforms=[_AddBatchSum()],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(unbatched_ds)


def test_post_batch_transforms_deferred_batching_via_replace(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  template_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      transforms=[_AddScalarPlusOne()],
      post_batch_transforms=[_AddBatchSum()],
  )
  batched_ds = dataclasses.replace(template_ds, batch_size=10)
  first_batch = next(iter(batched_ds))
  np.testing.assert_array_equal(
      first_batch['batch_sum'], np.full((10,), 55, dtype=np.int64)
  )


def test_post_batch_transforms_with_explicit_batch_transforms(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # Explicit `grain.TfBatch` in `transforms` with `batch_size=None`.
  explicit_batch_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      transforms=[
          _AddScalarPlusOne(),
          grain.TfBatch(batch_size=10, drop_remainder=True),
      ],
      post_batch_transforms=[_AddBatchSum()],
  )
  first_explicit_batch = next(iter(explicit_batch_ds))
  np.testing.assert_array_equal(
      first_explicit_batch['batch_sum'], np.full((10,), 55, dtype=np.int64)
  )

  # Custom `UnsafeTfDataTransform` (`padded_batch`) in `transforms`.
  padded_batch_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      transforms=[_AddScalarPlusOne(), _PaddedBatchTransform(batch_size=10)],
      post_batch_transforms=[_AddBatchSum()],
  )
  first_padded_batch = next(iter(padded_batch_ds))
  np.testing.assert_array_equal(
      first_padded_batch['batch_sum'], np.full((10,), 55, dtype=np.int64)
  )

  # `TfBatchAndPack` with odd `batch_size * seq_len` (3 * 5 = 15) + random op
  # across both `shuffle_after_pack=False` and `shuffle_after_pack=True`.
  for shuffle_after_pack in (False, True):

    def _make_packed_ds(seed: int, shuffle_pack: bool = shuffle_after_pack):
      return dummy_tfds_ds(
          data_dir=dummy_builder.data_dir_root,
          num_epochs=1,
          shuffle=False,
          seed=seed,
          batch_size=None,
          transforms=[
              lambda ex: ex | {'id': tf.expand_dims(ex['id'], axis=0)},
              grain.TfBatchAndPack(
                  batch_size=3,
                  sequence_lengths={'id': 5},
                  shuffle_after_pack=shuffle_pack,
              ),
          ],
          post_batch_transforms=[_BatchRandomTransform()],
      )

    packed_batches_0 = list(_make_packed_ds(seed=0))
    packed_batches_1 = list(_make_packed_ds(seed=1))
    # When `shuffle_after_pack=True`, `_seed` (len 2) is also packed into length
    # 5 (at most 2 examples/row -> 6/batch -> 17 batches; last batch has 4).
    expected_num_batches = 17 if shuffle_after_pack else 7
    expected_last_valid = 4 if shuffle_after_pack else 10
    assert len(packed_batches_0) == expected_num_batches
    assert len(packed_batches_1) == expected_num_batches
    assert (
        sum(np.count_nonzero(b['id_segment_ids'] > 0) for b in packed_batches_0)
        == 100
    )
    assert (
        np.count_nonzero(packed_batches_0[-1]['id_segment_ids'] > 0)
        == expected_last_valid
    )

    for b0, b1 in zip(packed_batches_0, packed_batches_1, strict=True):
      assert b0['post_rand'].shape == (3, 5)
      assert b1['post_rand'].shape == (3, 5)
      offsets_0 = b0['post_rand'] - b0['id'].astype(np.float32)
      offsets_1 = b1['post_rand'] - b1['id'].astype(np.float32)
      # `float32` ULP spacing up to `id = 99` is `2^-17` (~7.6e-6).
      np.testing.assert_allclose(offsets_0, offsets_0[0, 0], atol=1e-5)
      np.testing.assert_allclose(offsets_1, offsets_1[0, 0], atol=1e-5)
      assert 0.0 <= offsets_0[0, 0] < 1.0
      assert 0.0 <= offsets_1[0, 0] < 1.0
      assert abs(float(offsets_0[0, 0]) - float(offsets_1[0, 0])) > 1e-5
      assert not np.array_equal(b0['post_rand'], b1['post_rand'])


class _BatchFilterTransform(grain.FilterTransform):
  """Post-batch filter creating unknown cardinality."""

  def filter(self, features):
    return tf.reduce_any(features['id'] < 50)


class _DictRebuildingRandomTransform(grain.RandomMapTransform):
  """Post-batch random transform that rebuilds a fresh feature dictionary."""

  def random_map(self, features, seed):
    offset = tf.random.stateless_uniform([], seed)
    return {
        grain.INDEX: features[grain.INDEX],
        'id': features['id'],
        'rand_id': tf.cast(features['id'], tf.float32) + offset,
    }


class _UnbatchTransform(grain.UnsafeTfDataTransform):
  """Unbatches a batched dataset back to scalar elements."""

  def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.unbatch()


class _UnrankedIndexTransform(grain.UnsafeTfDataTransform):
  """Replaces `_index` with an unranked int64 tensor (`rank is None`)."""

  def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    def _erase_index_rank(ex):
      unranked_idx = tf.py_function(
          func=lambda x: x,
          inp=[ex[grain.INDEX]],
          Tout=tf.int64,
      )
      return dict(ex) | {grain.INDEX: unranked_idx}

    return dataset.map(_erase_index_rank)


def test_post_batch_transforms_preserves_padding_mask(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  keep_id = kd.data.Elements(keep=['id'])

  # Shared `Elements(keep=['id'])` preserves `__is_padding__` on PAD_WITH_MASK.
  masked_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:10]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[keep_id],
  )
  masked_batches = list(masked_ds)
  assert len(masked_batches) == 3
  assert set(masked_batches[-1].keys()) == {'id', '__is_padding__'}
  np.testing.assert_array_equal(masked_batches[-1]['id'], [8, 9, 0, 0])
  np.testing.assert_array_equal(
      masked_batches[-1]['__is_padding__'], [False, False, True, True]
  )

  # Same `keep_id` works without KeyError under DROP, KEEP, and PAD.
  for remainder_mode in (
      kd.data.tf.DropRemainder.DROP,
      kd.data.tf.DropRemainder.KEEP,
      kd.data.tf.DropRemainder.PAD,
  ):
    non_masked_ds = dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        split='train[:10]',
        num_epochs=1,
        shuffle=False,
        batch_size=4,
        batch_drop_remainder=remainder_mode,
        post_batch_transforms=[keep_id],
    )
    first_batch = next(iter(non_masked_ds))
    assert list(first_batch.keys()) == ['id']

  # Dict-rebuilding callables and RandomMapTransforms also preserve the mask.
  rebuilt_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:10]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[
          _DictRebuildingRandomTransform(),
          lambda ex: {'doubled': ex['id'] * 2, 'rand_id': ex['rand_id']},
      ],
  )
  rebuilt_batches = list(rebuilt_ds)
  assert set(rebuilt_batches[-1].keys()) == {
      'doubled',
      'rand_id',
      '__is_padding__',
  }
  np.testing.assert_array_equal(rebuilt_batches[-1]['doubled'], [16, 18, 0, 0])
  np.testing.assert_array_equal(
      rebuilt_batches[-1]['__is_padding__'], [False, False, True, True]
  )

  # Explicit `drop=['__is_padding__']` and `rename={'__is_padding__': ...}` are
  # respected without re-injecting `__is_padding__`.
  dropped_mask_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:10]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[kd.data.Elements(drop=['__is_padding__', 'rand'])],
  )
  assert list(next(iter(dropped_mask_ds)).keys()) == ['id']

  renamed_mask_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:10]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[
          kd.data.Elements(drop=['rand'], rename={'__is_padding__': 'pad_mask'})
      ],
  )
  renamed_last = list(renamed_mask_ds)[-1]
  assert set(renamed_last.keys()) == {'id', 'pad_mask'}
  np.testing.assert_array_equal(
      renamed_last['pad_mask'], [False, False, True, True]
  )


@pytest.mark.parametrize(
    'post_transform',
    [_BatchFilterTransform(), grain.IgnoreErrorsTransform()],
)
def test_multi_host_pad_with_mask_post_batch_unknown_cardinality_raises(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
    post_transform: grain.Transformation,
):  # pylint: disable=redefined-outer-name
  with (
      mock.patch('jax.process_count', return_value=2),
      mock.patch('jax.process_index', return_value=0),
      mock.patch(
          'jax.experimental.multihost_utils.process_allgather',
          return_value=np.array([10, 10]),
      ),
  ):
    ds = dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        split='train[:20]',
        num_epochs=1,
        shuffle=False,
        batch_size=8,
        batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
        post_batch_transforms=[post_transform],
    )
    with pytest.raises(
        ValueError,
        match='DropRemainder.PAD_WITH_MASK across multiple JAX processes',
    ):
      list(ds)


def test_post_batch_transforms_prebatched_zip_and_element_spec(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  ds_a = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=10,
      transforms=[_AddScalarPlusOne()],
  )
  ds_b = dummy_tfds_legacy_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=10,
      transforms=[_AddScalarPlusOne()],
  )
  prebatched_zip = kd.data.tf.ZipDatasets(
      datasets={'a': ds_a, 'b': ds_b},
      seed=0,
      batch_size=None,
      post_batch_transforms=[
          lambda ex: {'pair_sum': ex['a']['pre'] + ex['b']['pre']}
      ],
  )
  first_zip_batch = next(iter(prebatched_zip))
  np.testing.assert_array_equal(
      first_zip_batch['pair_sum'], np.arange(2, 22, 2, dtype=np.int64)
  )

  spec_ds = kd.data.tf.ElementSpecDataset(
      spec={'x': enp.ArraySpec(shape=(4, 8), dtype=np.float32)},
      seed=0,
      post_batch_transforms=[lambda ex: {'y': ex['x'] + 3.0}],
  )
  spec_batch = next(iter(spec_ds))
  np.testing.assert_array_equal(
      spec_batch['y'], np.full((4, 8), 3.0, dtype=np.float32)
  )


def test_post_batch_transforms_rejects_unbatched_sources_and_unbatching(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # Unbatched ZipDatasets (even with multi-dimensional image leaves) raises.
  unbatched_img_child = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      transforms=[lambda ex: {'img': tf.zeros((4, 4, 3), dtype=tf.float32)}],
  )
  unbatched_zip = kd.data.tf.ZipDatasets(
      datasets={'a': unbatched_img_child},
      seed=0,
      batch_size=None,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(unbatched_zip)

  # Scalar ElementSpecDataset raises.
  scalar_spec_ds = kd.data.tf.ElementSpecDataset(
      spec={'id': enp.ArraySpec(shape=(), dtype=np.int64)},
      seed=0,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(scalar_spec_ds)

  # Pipeline that batches and then unbatches in `transforms` raises.
  unbatched_after_batch = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      transforms=[
          _AddScalarPlusOne(),
          grain.TfBatch(batch_size=10, drop_remainder=True),
          _UnbatchTransform(),
      ],
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(unbatched_after_batch)

  # Pipeline with unranked `_index` (`rank is None`) raises.
  unranked_index_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      transforms=[_UnrankedIndexTransform()],
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(unranked_index_ds)


def test_post_batch_random_transforms_asymmetric_mixture_and_rebatch(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # Pre-batched child with post-batch RandomMapTransform mixed with pre-batched
  # sibling without RandomMapTransform keeps matching `_seed` specs.
  child_with_rand = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=10,
      transforms=[],
      post_batch_transforms=[_BatchRandomTransform(out_key='val')],
  )
  child_without_rand = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=10,
      transforms=[],
      post_batch_transforms=[
          lambda ex: ex | {'val': tf.cast(ex['id'], tf.float32)}
      ],
  )
  asymmetric_mix = kd.data.tf.SampleFromDatasets(
      [child_with_rand, child_without_rand],
      stop_on_empty_dataset=True,
      seed=0,
      post_batch_transforms=[
          lambda ex: ex | {'val2': ex['val'] + 1.0},
      ],
  )
  mix_batch = next(iter(asymmetric_mix))
  assert mix_batch['val2'].shape == (10,)

  # Re-batching between two post-batch RandomMapTransforms works cleanly.
  rebatched_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      seed=0,
      batch_size=5,
      transforms=[],
      post_batch_transforms=[
          _BatchRandomTransform(out_key='r1'),
          grain.TfBatch(batch_size=2, drop_remainder=True),
          _BatchRandomTransform(out_key='r2'),
      ],
  )
  rebatched = list(rebatched_ds)
  assert len(rebatched) == 10
  assert rebatched[0]['r1'].shape == (2, 5)
  assert rebatched[0]['r2'].shape == (2, 5)
  assert not np.array_equal(rebatched[0]['r1'], rebatched[0]['r2'])


def test_post_batch_random_transforms_tf_batch_and_pack_empty_first_row(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # 5 tokens packed into `(batch_size=8, seq_len=5)` produces 1 valid row and
  # 7 empty padding rows, ensuring row 0 is empty on most shuffles and testing
  # seed sensitivity even when row 0 has zeroed `_seed`.
  def _run_single_partial_pack(seed: int):
    ds = dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        split='train[:5]',
        num_epochs=1,
        shuffle=False,
        seed=seed,
        batch_size=None,
        transforms=[
            lambda ex: ex | {'id': tf.expand_dims(ex['id'], axis=0)},
            grain.TfBatchAndPack(
                batch_size=8,
                sequence_lengths={'id': 5},
                shuffle_after_pack=True,
            ),
        ],
        post_batch_transforms=[_BatchRandomTransform()],
    )
    (batch,) = list(ds)
    return batch

  empty_row0_offsets = set()
  for seed in range(10):
    batch = _run_single_partial_pack(seed)
    if np.all(batch['id_segment_ids'][0] == 0):
      offset = float(batch['post_rand'][0, 0] - batch['id'][0, 0])
      empty_row0_offsets.add(offset)
  # Multiple seeds land an empty row at index 0 and each still gets a distinct
  # random offset (never stuck on a constant `[0, 0]` seed).
  assert len(empty_row0_offsets) >= 2


def test_multi_host_pad_with_mask_explicit_batch_and_valid_post_batch(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # Explicit `TfBatch` with `batch_size=None` and `PAD_WITH_MASK` across hosts:
  # uneven host cardinalities (e.g. [4, 4, 3, 3]) must raise.
  with (
      mock.patch('jax.process_count', return_value=4),
      mock.patch('jax.process_index', return_value=0),
      mock.patch(
          'jax.experimental.multihost_utils.process_allgather',
          return_value=np.array([4, 4, 3, 3]),
      ),
  ):
    uneven_ds = dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        split='train[:98]',
        num_epochs=1,
        shuffle=False,
        batch_size=None,
        batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
        transforms=[grain.TfBatch(batch_size=8, drop_remainder=False)],
        post_batch_transforms=[lambda ex: ex | {'id2': ex['id'] * 2}],
    )
    with pytest.raises(
        ValueError,
        match='requires equal dataset cardinality across hosts',
    ):
      _ = list(uneven_ds)

  # Even when host batch counts match, missing `__is_padding__` raises.
  with (
      mock.patch('jax.process_count', return_value=4),
      mock.patch('jax.process_index', return_value=0),
      mock.patch(
          'jax.experimental.multihost_utils.process_allgather',
          return_value=np.array([3, 3, 3, 3]),
      ),
  ):
    missing_mask_ds = dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        split='train[:92]',
        num_epochs=1,
        shuffle=False,
        batch_size=None,
        batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
        transforms=[grain.TfBatch(batch_size=8, drop_remainder=False)],
        post_batch_transforms=[lambda ex: ex | {'id2': ex['id'] * 2}],
    )
    with pytest.raises(
        ValueError,
        match="requires '__is_padding__' in `element_spec`",
    ):
      _ = list(missing_mask_ds)

  # Standard `batch_size=32` multi-host `PAD_WITH_MASK` succeeds both when
  # keeping `__is_padding__` and when explicitly dropping or renaming it.
  for post_tr, expected_keys in (
      (
          [kd.data.Elements(keep=['id'])],
          {'id', '__is_padding__'},
      ),
      (
          [kd.data.Elements(drop=['__is_padding__', 'rand'])],
          {'id'},
      ),
      (
          [
              kd.data.Elements(
                  drop=['rand'], rename={'__is_padding__': 'pad_mask'}
              )
          ],
          {'id', 'pad_mask'},
      ),
  ):
    with (
        mock.patch('jax.process_count', return_value=4),
        mock.patch('jax.process_index', return_value=0),
        mock.patch(
            'jax.experimental.multihost_utils.process_allgather',
            side_effect=[
                np.array([25, 25, 24, 24]),
                np.array([4, 4, 4, 4]),
            ],
        ),
    ):
      valid_ds = dummy_tfds_ds(
          data_dir=dummy_builder.data_dir_root,
          split='train[:98]',
          num_epochs=1,
          shuffle=False,
          batch_size=32,
          batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
          post_batch_transforms=post_tr,
      )
      batches = list(valid_ds)
      assert len(batches) == 4
      assert set(batches[0].keys()) == expected_keys


def test_mixture_batch_size_none_preserves_explicit_child_drop_remainder(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # On `train[:98]` across 4 hosts, `DropRemainder.DROP` shards with
  # `drop_remainder=True` (24 records/host -> 3 full batches of 8), whereas
  # `PAD_WITH_MASK` disables shard `drop_remainder` (25 records on host 0).
  with (
      mock.patch('jax.process_count', return_value=4),
      mock.patch('jax.process_index', return_value=0),
  ):
    explicit_child = dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        split='train[:98]',
        num_epochs=1,
        shuffle=False,
        batch_size=None,
        batch_drop_remainder=kd.data.tf.DropRemainder.DROP,
        transforms=[grain.TfBatch(batch_size=8, drop_remainder=False)],
    )
    sample_mix = kd.data.tf.SampleFromDatasets(
        [explicit_child],
        seed=0,
        batch_size=None,
        batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
    )
    sample_batches = list(sample_mix)
    assert len(sample_batches) == 3
    assert [b['id'].shape for b in sample_batches] == [(8,), (8,), (8,)]

    zip_mix = kd.data.tf.ZipDatasets(
        {'a': explicit_child},
        seed=0,
        batch_size=None,
        batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
    )
    zip_batches = list(zip_mix)
    assert len(zip_batches) == 3
    assert [b['a']['id'].shape for b in zip_batches] == [(8,), (8,), (8,)]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _CustomPrebatchedPipeline(kd.data.tf.TFDataPipeline):
  """Custom pre-batched `TFDataPipeline` subclass for testing `is_prebatched`."""

  dataset_fn: object
  is_prebatched: bool = True
  _supports_symbolic_checkpoint: ClassVar[bool] = False  # pyrefly: ignore[bad-override]

  def ds_for_current_process(self, rng):
    del rng
    return self.dataset_fn()  # pyrefly: ignore[not-callable]


class _EmptyIndexTransform(grain.UnsafeTfDataTransform):
  """Replaces `_index` with an empty 1D `shape=(0,)` dummy tensor."""

  def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(
        lambda ex: dict(ex) | {grain.INDEX: tf.constant((), dtype=tf.int64)}
    )


def test_prebatched_custom_subclasses_and_spec_validation(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # Custom pre-batched subclass with RaggedTensor, SparseTensor, and unranked
  # `py_function` leaves works with `post_batch_transforms` and random ops.
  def _make_composite_ds():
    def _build_batch(_):
      ragged = tf.ragged.constant([[1, 2], [3], [4, 5, 6], [7]], dtype=tf.int32)
      sparse = tf.SparseTensor(
          indices=[[0, 0], [2, 1], [3, 0]],
          values=tf.constant([10, 20, 30], dtype=tf.int32),
          dense_shape=[4, 3],
      )
      unranked = tf.py_function(
          func=lambda: np.arange(4, dtype=np.int64),
          inp=[],
          Tout=tf.int64,
      )
      return {
          'id': tf.constant([0, 1, 2, 3], dtype=tf.int64),
          'ragged': ragged,
          'sparse': sparse,
          'unranked': unranked,
      }

    return tf.data.Dataset.range(2).map(_build_batch)

  composite_ds = _CustomPrebatchedPipeline(
      dataset_fn=_make_composite_ds,
      seed=0,
      post_batch_transforms=[
          _BatchRandomTransform(out_key='post_rand'),
          lambda ex: {
              'id': ex['id'],
              'post_rand': ex['post_rand'],
              'row_lengths': ex['ragged'].row_lengths(),
              'sparse_dense': tf.sparse.to_dense(ex['sparse']),
              'unranked': ex['unranked'],
          },
      ],
  )
  composite_batches = list(composite_ds)
  assert len(composite_batches) == 2
  np.testing.assert_array_equal(
      composite_batches[0]['row_lengths'], [2, 1, 3, 1]
  )
  np.testing.assert_array_equal(composite_batches[0]['unranked'], [0, 1, 2, 3])
  assert composite_batches[0]['sparse_dense'].shape == (4, 3)
  assert not np.array_equal(
      composite_batches[0]['post_rand'], composite_batches[1]['post_rand']
  )

  # Invalid pre-batched specs (scalar leaf, zero batch dim, mismatched batch
  # dims, or unbatched Grain source with `is_prebatched=True`) raise ValueError.
  scalar_leaf_ds = _CustomPrebatchedPipeline(
      dataset_fn=lambda: tf.data.Dataset.from_tensors(
          {'x': tf.zeros((4, 2)), 'scalar': tf.constant(1)}
      ),
      seed=0,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(ValueError, match='rank >= 1'):
    _ = iter(scalar_leaf_ds)

  zero_dim_ds = _CustomPrebatchedPipeline(
      dataset_fn=lambda: tf.data.Dataset.from_tensors({'x': tf.zeros((0, 2))}),
      seed=0,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(ValueError, match='non-zero leading batch dimension'):
    _ = iter(zero_dim_ds)

  mismatched_dim_ds = _CustomPrebatchedPipeline(
      dataset_fn=lambda: tf.data.Dataset.from_tensors(
          {'x': tf.zeros((4, 2)), 'y': tf.zeros((8, 2))}
      ),
      seed=0,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(ValueError, match='matching leading batch dimensions'):
    _ = iter(mismatched_dim_ds)

  unbatched_grain_marked_prebatched = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      is_prebatched=True,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(ValueError, match='rank >= 1'):
    _ = iter(unbatched_grain_marked_prebatched)

  # Runtime validation catches `py_function` leaves returning scalar or
  # mismatched batch sizes when `is_prebatched=True`.
  bad_unranked_scalar_ds = _CustomPrebatchedPipeline(
      dataset_fn=lambda: tf.data.Dataset.range(1).map(
          lambda _: {
              'u': tf.py_function(
                  func=lambda: np.int64(7), inp=[], Tout=tf.int64
              )
          }
      ),
      seed=0,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(tf.errors.InvalidArgumentError, match='rank >= 1'):
    _ = list(bad_unranked_scalar_ds)

  bad_unranked_mismatch_ds = _CustomPrebatchedPipeline(
      dataset_fn=lambda: tf.data.Dataset.range(1).map(
          lambda _: {
              'x': tf.zeros((4, 2), dtype=tf.float32),
              'u': tf.py_function(
                  func=lambda: np.arange(8, dtype=np.int64),
                  inp=[],
                  Tout=tf.int64,
              ),
          }
      ),
      seed=0,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      tf.errors.InvalidArgumentError,
      match='matching leading batch dimensions',
  ):
    _ = list(bad_unranked_mismatch_ds)

  # Pre-batched ElementSpecDataset mixes cleanly with batched Tfds in
  # SampleFromDatasets because Grain meta features share matching batch shapes.
  tfds_batched = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      transforms=[kd.data.Elements(keep=['id'])],
  )
  spec_batched = kd.data.tf.ElementSpecDataset(
      spec={'id': enp.ArraySpec(shape=(4,), dtype=np.int64)},
      seed=0,
  )
  mixed_prebatched = kd.data.tf.SampleFromDatasets(
      [tfds_batched, spec_batched],
      stop_on_empty_dataset=True,
      seed=0,
      post_batch_transforms=[lambda ex: {'id2': ex['id'] + 1}],
  )
  assert next(iter(mixed_prebatched))['id2'].shape == (4,)


def test_mixed_rank_spec_and_unbatch_rejected_everywhere(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # Unbatched ElementSpecDataset with 3D image + scalar label is rejected.
  mixed_rank_spec_ds = kd.data.tf.ElementSpecDataset(
      spec={
          'image': enp.ArraySpec(shape=(8, 8, 3), dtype=np.float32),
          'label': enp.ArraySpec(shape=(), dtype=np.int32),
      },
      seed=0,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(mixed_rank_spec_ds)

  # Explicit `is_prebatched=False` overrides auto-detection for single 3D image.
  single_img_unbatched = kd.data.tf.ElementSpecDataset(
      spec={'image': enp.ArraySpec(shape=(8, 8, 3), dtype=np.float32)},
      is_prebatched=False,
      seed=0,
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(single_img_unbatched)

  # `.unbatch()` in `transforms` collapses `_index` to scalar `()` and is caught
  # on ElementSpecDataset, pre-batched ZipDatasets, SampleFromDatasets, and
  # custom `is_prebatched=True` pipelines even when unbatched leaves stay 3D.
  unbatched_spec_ds = kd.data.tf.ElementSpecDataset(
      spec={
          'image': enp.ArraySpec(shape=(4, 8, 8, 3), dtype=np.float32),
          'label': enp.ArraySpec(shape=(4,), dtype=np.int32),
      },
      seed=0,
      transforms=[_UnbatchTransform()],
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(unbatched_spec_ds)

  batched_img_child = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      transforms=[lambda ex: {'image': tf.zeros((8, 8, 3), dtype=tf.float32)}],
  )
  unbatched_zip = kd.data.tf.ZipDatasets(
      datasets={'a': batched_img_child},
      seed=0,
      batch_size=None,
      transforms=[_UnbatchTransform()],
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(unbatched_zip)

  unbatched_sample = kd.data.tf.SampleFromDatasets(
      [batched_img_child],
      seed=0,
      batch_size=None,
      transforms=[_UnbatchTransform()],
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(unbatched_sample)

  unbatched_custom = _CustomPrebatchedPipeline(
      dataset_fn=lambda: tf.data.Dataset.from_tensors(
          {'image': tf.zeros((4, 8, 8, 3), dtype=tf.float32)}
      ),
      seed=0,
      transforms=[_UnbatchTransform()],
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(unbatched_custom)

  # Empty `shape=(0,)` dummy `_index` is rejected.
  empty_idx_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      transforms=[_EmptyIndexTransform()],
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(empty_idx_ds)


@dataclasses.dataclass(frozen=True)
class _CluRandomOp:
  """CLU `PreprocessOp` splitting `SEED_KEY` inside `PreprocessFn`."""

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    features = dict(features)
    raw_seed = features.pop(preprocess_spec.SEED_KEY)
    assert isinstance(raw_seed, tf.Tensor)
    next_seed, seed = tf.unstack(
        tf.random.experimental.stateless_split(raw_seed)
    )
    offset = tf.random.stateless_uniform([], seed)
    features['clu_rand'] = tf.cast(features['id'], tf.float32) + offset
    features[preprocess_spec.SEED_KEY] = next_seed
    return features


@dataclasses.dataclass(frozen=True)
class _CluAddRandomOffset(preprocess_spec.RandomMapTransform):
  """Standalone CLU `RandomMapTransform` for testing post-batch wrapping."""

  def _transform(self, features, seed):
    offset = tf.random.stateless_uniform([], seed)
    features = dict(features)
    features['clu_rand2'] = tf.cast(features['id'], tf.float32) + offset
    return features


@dataclasses.dataclass(frozen=True)
class _CluKeepKeys:
  """CLU `PreprocessOp` keeping only specified keys."""

  keys: tuple[str, ...]

  def __call__(
      self, features: preprocess_spec.Features
  ) -> preprocess_spec.Features:
    return {k: v for k, v in features.items() if k in self.keys}


def test_post_batch_padding_mask_validation_clu_and_prebatched_zip(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # 1. Changing axis 0 (`B -> 2*B`) without updating `__is_padding__` raises
  # both when `__is_padding__` is re-attached and when kept as `(B,)`.
  for bad_tr in (
      lambda b: {'id': tf.concat([b['id'], b['id']], axis=0)},
      lambda b: {
          'id': tf.concat([b['id'], b['id']], axis=0),
          '__is_padding__': b['__is_padding__'],
      },
      kd.data.tf.OneHot(key='id', num_classes=10),  # multi=True -> shape (10,)
      grain.TfBatch(batch_size=2, drop_remainder=True),  # rank 2 mask (2, 4)
  ):
    bad_axis_ds = dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        split='train[:10]',
        num_epochs=1,
        shuffle=False,
        batch_size=4,
        batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
        post_batch_transforms=[bad_tr],
    )
    with pytest.raises(ValueError, match='__is_padding__'):
      _ = list(bad_axis_ds)

  # `OneHot(..., multi=False)` keeps leading dim `(4, 10)` and succeeds.
  onehot_ok_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:10]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[
          kd.data.tf.OneHot(key='id', num_classes=12, multi=False)
      ],
  )
  onehot_batches = list(onehot_ok_ds)
  assert onehot_batches[-1]['id'].shape == (4, 12)
  np.testing.assert_array_equal(
      onehot_batches[-1]['__is_padding__'], [False, False, True, True]
  )

  # Updating `__is_padding__` alongside a `B -> 2*B` transform succeeds.
  doubled_ok_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:6]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[
          lambda b: {
              'id': tf.concat([b['id'], b['id']], axis=0),
              '__is_padding__': tf.concat(
                  [b['__is_padding__'], b['__is_padding__']], axis=0
              ),
          }
      ],
  )
  doubled_last = list(doubled_ok_ds)[-1]
  assert doubled_last['id'].shape == (8,)
  np.testing.assert_array_equal(
      doubled_last['__is_padding__'],
      [False, False, True, True, False, False, True, True],
  )

  # 2. Dict-wide cast corrupting `__is_padding__` dtype raises ValueError.
  bad_dtype_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:10]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[
          lambda b: jax.tree.map(lambda x: tf.cast(x, tf.bfloat16), b)
      ],
  )
  with pytest.raises(ValueError, match='dtype tf.bool'):
    _ = list(bad_dtype_ds)

  # Dynamic axis-0 change via `tf.py_function` is caught at runtime.
  bad_dynamic_axis_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:6]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[
          lambda b: {
              'id': tf.py_function(
                  func=lambda x: np.concatenate([x, x], axis=0),
                  inp=[b['id']],
                  Tout=tf.int64,
              )
          }
      ],
  )
  with pytest.raises(
      tf.errors.InvalidArgumentError,
      match='__is_padding__',
  ):
    _ = list(bad_dynamic_axis_ds)

  # 3. `clu.preprocess_spec.PreprocessFn` (and `RandomMapTransform`) get `[2]`
  # `_seed` and preserve `__is_padding__` even when `Keep` drops unlisted keys.
  clu_fn = preprocess_spec.PreprocessFn(
      ops=[_CluRandomOp(), _CluKeepKeys(keys=('id', 'clu_rand'))],
      only_jax_types=True,
  )

  def _make_clu_ds(seed: int):
    return dummy_tfds_ds(
        data_dir=dummy_builder.data_dir_root,
        split='train[:10]',
        num_epochs=1,
        shuffle=False,
        seed=seed,
        batch_size=4,
        batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
        post_batch_transforms=[clu_fn, _CluAddRandomOffset()],
    )

  clu_batches_0a = list(_make_clu_ds(seed=0))
  clu_batches_0b = list(_make_clu_ds(seed=0))
  clu_batches_1 = list(_make_clu_ds(seed=1))
  assert len(clu_batches_0a) == 3
  assert set(clu_batches_0a[-1].keys()) == {
      'id',
      'clu_rand',
      'clu_rand2',
      '__is_padding__',
  }
  np.testing.assert_array_equal(
      clu_batches_0a[-1]['__is_padding__'], [False, False, True, True]
  )
  np.testing.assert_array_equal(
      clu_batches_0a[0]['clu_rand'], clu_batches_0b[0]['clu_rand']
  )
  assert not np.array_equal(
      clu_batches_0a[0]['clu_rand'], clu_batches_1[0]['clu_rand']
  )

  # 4. Pre-batched `ZipDatasets` pops child `__is_padding__` masks, hoists
  # their `logical_or` to top-level `__is_padding__`, and preserves it across
  # dict-rebuilding `post_batch_transforms`.
  child_a = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:10]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
  )
  child_b = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:11]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
  )
  raw_zip = kd.data.tf.ZipDatasets(
      datasets={'a': child_a, 'b': child_b},
      seed=0,
      batch_size=None,
  )
  raw_zip_last = list(raw_zip)[-1]
  assert '__is_padding__' not in raw_zip_last['a']
  assert '__is_padding__' not in raw_zip_last['b']
  # child_a mask: [F, F, T, T], child_b mask: [F, F, F, T] -> OR: [F, F, T, T].
  np.testing.assert_array_equal(
      raw_zip_last['__is_padding__'], [False, False, True, True]
  )

  rebuilt_zip = kd.data.tf.ZipDatasets(
      datasets={'a': child_a, 'b': child_b},
      seed=0,
      batch_size=None,
      post_batch_transforms=[
          lambda ex: {'pair_sum': ex['a']['id'] + ex['b']['id']}
      ],
  )
  rebuilt_zip_last = list(rebuilt_zip)[-1]
  assert set(rebuilt_zip_last.keys()) == {'pair_sum', '__is_padding__'}
  np.testing.assert_array_equal(rebuilt_zip_last['pair_sum'], [16, 18, 10, 0])
  np.testing.assert_array_equal(
      rebuilt_zip_last['__is_padding__'], [False, False, True, True]
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class _UnannotatedPrebatchedPipeline(kd.data.tf.TFDataPipeline):
  """Subclass setting unannotated `is_prebatched = True` with dynamic dim0."""

  dataset_fn: object
  is_prebatched = True
  _supports_symbolic_checkpoint: ClassVar[bool] = False  # pyrefly: ignore[bad-override]

  def ds_for_current_process(self, rng):
    del rng
    return self.dataset_fn()  # pyrefly: ignore[not-callable]


def test_dynamic_prebatched_and_zip_outer_batch_edge_cases(
    dummy_builder: tfds.core.GeneratorBasedBuilder,
):  # pylint: disable=redefined-outer-name
  # 1. Unannotated `is_prebatched = True` on a dynamic-batch-size (`(None,)`)
  # dataset attaches dynamic 1D `_index` / 2D `_seed` and supports `.unbatch()`.
  def _make_dynamic_batches():
    return tf.data.Dataset.range(6).map(lambda x: {'id': x}).batch(4)

  dyn_prebatched = _UnannotatedPrebatchedPipeline(
      dataset_fn=_make_dynamic_batches,
      seed=0,
      post_batch_transforms=[_BatchRandomTransform(out_key='post_rand')],
  )
  dyn_batches = list(dyn_prebatched)
  assert [b['id'].shape for b in dyn_batches] == [(4,), (2,)]
  assert [b['post_rand'].shape for b in dyn_batches] == [(4,), (2,)]

  dyn_unbatched = _UnannotatedPrebatchedPipeline(
      dataset_fn=_make_dynamic_batches,
      seed=0,
      transforms=[_UnbatchTransform()],
      post_batch_transforms=[lambda ex: ex],
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(dyn_unbatched)

  # 2. `ZipDatasets` with outer `batch_size` over already-batched children keeps
  # pre-outer-batch `_seed` at rank 1 `[2]` for `transforms` and batches to
  # `(outer_batch, child_batch)` for `post_batch_transforms`.
  batched_child = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:16]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
  )
  rebatched_zip = kd.data.tf.ZipDatasets(
      datasets={'a': batched_child},
      seed=0,
      batch_size=2,
      transforms=[
          lambda ex: {'id': ex['a']['id']},
          _BatchRandomTransform(out_key='pre_outer_rand'),
      ],
      post_batch_transforms=[_BatchRandomTransform(out_key='post_outer_rand')],
  )
  rebatched_zip_batches = list(rebatched_zip)
  assert len(rebatched_zip_batches) == 2
  assert rebatched_zip_batches[0]['pre_outer_rand'].shape == (2, 4)
  assert rebatched_zip_batches[0]['post_outer_rand'].shape == (2, 4)

  # 3. `ZipDatasets(batch_size=None)` over children with different batch sizes
  # still iterates when `post_batch_transforms` is empty, and raises when set.
  child_4 = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:8]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
  )
  child_8 = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:16]',
      num_epochs=1,
      shuffle=False,
      batch_size=8,
  )
  hetero_zip = kd.data.tf.ZipDatasets(
      datasets={'a': child_4, 'b': child_8},
      seed=0,
      batch_size=None,
  )
  hetero_batch = next(iter(hetero_zip))
  assert hetero_batch['a']['id'].shape == (4,)
  assert hetero_batch['b']['id'].shape == (8,)

  hetero_zip_post = dataclasses.replace(
      hetero_zip, post_batch_transforms=[lambda ex: ex]
  )
  with pytest.raises(
      ValueError,
      match='Cannot apply `post_batch_transforms` when batching is disabled',
  ):
    _ = iter(hetero_zip_post)

  # 4. Dynamic `(None,)` `__is_padding__` still statically catches mismatched
  # static output feature batch dimensions (`(4,)` vs `(8,)`).
  dyn_mask_mismatch_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      split='train[:8]',
      num_epochs=1,
      shuffle=False,
      batch_size=4,
      batch_drop_remainder=kd.data.tf.DropRemainder.PAD_WITH_MASK,
      post_batch_transforms=[
          lambda b: {
              'a': b['id'],
              'b': tf.concat([b['id'], b['id']], axis=0),
              '__is_padding__': tf.boolean_mask(
                  b['__is_padding__'], tf.ones_like(b['__is_padding__'])
              ),
          }
      ],
  )
  with pytest.raises(ValueError, match='__is_padding__'):
    _ = iter(dyn_mask_mismatch_ds)
