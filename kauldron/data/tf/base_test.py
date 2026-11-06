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

  # `TfBatchAndPack` with odd `batch_size * seq_len` (3 * 5 = 15) + random op.
  packed_ds = dummy_tfds_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=False,
      batch_size=None,
      transforms=[
          lambda ex: ex | {'id': tf.expand_dims(ex['id'], axis=0)},
          grain.TfBatchAndPack(
              batch_size=3,
              sequence_lengths={'id': 5},
              shuffle_after_pack=True,
          ),
      ],
      post_batch_transforms=[_BatchRandomTransform()],
  )
  first_packed_batch = next(iter(packed_ds))
  assert first_packed_batch['post_rand'].shape == (3, 5)
