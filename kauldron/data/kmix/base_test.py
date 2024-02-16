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

from unittest import mock

from kauldron.data import kmix
import numpy as np
import pytest
import tensorflow_datasets as tfds


# TODO(epot): Parametrize over Tfds, SeqIO,...
# Test:
# * Deterministic random transforms (even when shuffle == False)
# * Test mixtures


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


def _ids(ds) -> list[int]:
  return [ex['id'] for ex in ds]


@pytest.fixture(scope='module')
def dummy_builder(tmp_path_factory: pytest.TempPathFactory):
  tmp_path = tmp_path_factory.mktemp('data_dir')
  builder = DummyDataset(data_dir=tmp_path)
  builder.download_and_prepare(file_format='array_record')
  yield builder


def test_no_shuffle(dummy_builder: tfds.core.GeneratorBasedBuilder):  # pylint: disable=redefined-outer-name
  ds = kmix.Tfds(  # pytype: disable=wrong-keyword-args
      name='dummy_dataset',
      split='train',
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=False,
      seed=0,
  )
  assert len(ds) == 200
  assert list(ds.element_spec.keys()) == ['id']  # pytype: disable=attribute-error
  assert _ids(ds) == list(range(100)) + list(range(100))


def test_shuffle(dummy_builder: tfds.core.GeneratorBasedBuilder):  # pylint: disable=redefined-outer-name
  ds = kmix.Tfds(  # pytype: disable=wrong-keyword-args
      name='dummy_dataset',
      split='train',
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=True,
      seed=0,
  )
  assert len(ds) == 100

  # Iterating twice on the dataset yields the same results.
  ids = _ids(ds)
  assert _ids(ds) == ids

  ds = kmix.Tfds(  # pytype: disable=wrong-keyword-args
      name='dummy_dataset',
      split='train',
      data_dir=dummy_builder.data_dir_root,
      num_epochs=2,
      shuffle=True,
      seed=0,
  )
  ids2 = _ids(ds)
  assert len(ids2) == 200
  assert ids2[:100] == ids  # First epoch is the same
  assert ids2 != ids + ids  # The second epoch has different shuffling.

  ds = kmix.Tfds(  # pytype: disable=wrong-keyword-args
      name='dummy_dataset',
      split='train',
      data_dir=dummy_builder.data_dir_root,
      num_epochs=1,
      shuffle=True,
      seed=1,  # Different seed
  )
  assert _ids(ds) != ids  # Seed change the shuffling


def test_sharding(dummy_builder: tfds.core.GeneratorBasedBuilder):  # pylint: disable=redefined-outer-name
  all_ids = set()
  with mock.patch('jax.process_count', return_value=4):
    for process_index in range(4):
      with mock.patch('jax.process_index', return_value=process_index):
        ds = kmix.Tfds(  # pytype: disable=wrong-keyword-args
            name='dummy_dataset',
            split='train',
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
