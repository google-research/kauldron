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

"""Tests."""

import dataclasses
import functools
from unittest import mock

import kauldron as kd
from kauldron.data import kmix
import numpy as np
import pytest
import tensorflow_datasets as tfds


# TODO(epot): Test:
# * Deterministic random transforms (even when shuffle == False)
dummy_tfds_ds = functools.partial(
    kmix.Tfds,
    name='dummy_dataset',
    split='train',
    seed=0,
)


dummy_tfds_legacy_ds = functools.partial(
    kmix.TfdsLegacy,
    name='dummy_dataset',
    split='train',
    seed=0,
    shuffle_buffer_size=100,
)


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


@pytest.fixture(scope='module')
def dummy_builder(tmp_path_factory: pytest.TempPathFactory):
  tmp_path = tmp_path_factory.mktemp('data_dir')
  builder = DummyDataset(data_dir=tmp_path)
  builder.download_and_prepare(file_format='array_record')
  yield builder


@with_ds_cls
def test_no_shuffle(
    ds_cls: type[kmix.Base], dummy_builder: tfds.core.GeneratorBasedBuilder
):  # pylint: disable=redefined-outer-name
  ds = ds_cls(  # pytype: disable=wrong-keyword-args
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=False,
  )
  assert len(ds) == 200
  assert list(ds.element_spec.keys()) == ['id']  # pytype: disable=attribute-error
  assert _ids(ds) == list(range(100)) + list(range(100))


@with_ds_cls
def test_shuffle(
    ds_cls: type[kmix.Base], dummy_builder: tfds.core.GeneratorBasedBuilder
):  # pylint: disable=redefined-outer-name
  ds = ds_cls(  # pytype: disable=wrong-keyword-args
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=True,
  )
  assert len(ds) == 100

  # Iterating twice on the dataset yields the same results.
  ids = _ids(ds)
  if isinstance(ds, kmix.WithShuffleBuffer):
    # When `ds.shuffle` is used, the dataset has to be re-created as
    # `reshuffle_each_iteration` is persistent on the cached iterator.
    ds = dataclasses.replace(ds)
  assert _ids(ds) == ids

  ds = ds_cls(  # pytype: disable=wrong-keyword-args
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=True,
  )
  ids2 = _ids(ds)
  assert len(ids2) == 200
  assert ids2[:100] == ids  # First epoch is the same
  assert ids2 != ids + ids  # The second epoch has different shuffling.

  ds = ds_cls(  # pytype: disable=wrong-keyword-args
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=True,
      seed=1,  # Different seed
  )
  ids_new_seed = _ids(ds)
  assert len(ids_new_seed) == len(ids)
  assert ids_new_seed != ids  # Seed change the shuffling


@with_ds_cls
def test_sharding(
    ds_cls: type[kmix.Base], dummy_builder: tfds.core.GeneratorBasedBuilder
):  # pylint: disable=redefined-outer-name
  all_ids = set()
  with mock.patch('jax.process_count', return_value=4):
    for process_index in range(4):
      with mock.patch('jax.process_index', return_value=process_index):
        ds = ds_cls(  # pytype: disable=wrong-keyword-args
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
          kd.data.Elements(rename={'id': 'code'}),
      ],
  )
  ds2 = dummy_tfds_legacy_ds(
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=False,
      transforms=[
          kd.data.Elements(rename={'id': 'code'}),
      ],
  )
  dsmix = kmix.SampleFromDatasets(  # pytype: disable=wrong-keyword-args
      [ds1, ds2], seed=0
  )
  ids = _ids(dsmix, key='code')
  assert len(ids) == 300
  assert sorted(ids) == sorted(list(range(100)) * 3)
