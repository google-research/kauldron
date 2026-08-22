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

"""Tests for data sources."""

import contextlib
from unittest import mock
from etils import enp
from grain import python as grain
import jax
from kauldron import kd
import numpy as np
import pytest
import tensorflow_datasets as tfds


def test_tfds():
  num_examples = 6
  num_epochs = 3
  batch_size = 2

  ds = kd.data.py.Tfds(  # pylint: disable=wrong-keyword-args
      name='mnist',
      split='train',
      shuffle=True,
      transforms=[
          kd.data.py.ValueRange(
              key='image',
              vrange=(0.0, 1.0),
          ),
      ],
      batch_size=batch_size,
      seed=0,
      num_epochs=num_epochs,
      num_workers=0,
  )

  with tfds.testing.mock_data(num_examples=num_examples):
    (ex,) = ds.take(1)
    assert set(ex.keys()) == {'image', 'label'}
    assert ex['image'].shape == (2, 28, 28, 1)
    # Here `num_examples % batch_size == 0`
    assert len(ds) == num_examples * num_epochs / batch_size
    # Check element_spec
    assert ds.element_spec == {
        'image': enp.ArraySpec(shape=(2, 28, 28, 1), dtype=np.float32),
        'label': enp.ArraySpec(shape=(2,), dtype=np.int64),
    }


@pytest.mark.parametrize('num_workers', [0, 2])
def test_range(num_workers: int):
  ds = kd.data.py.DataSource(
      grain.RangeDataSource(0, 10, 1),
      shuffle=False,
      num_workers=num_workers,
  )
  assert list(ds.take(10)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@pytest.mark.parametrize(
    'batch_drop_remainder,expected_output,should_raise',
    [
        pytest.param(True, [[0, 1, 2], [3, 4, 5], [6, 7, 8]], False, id='True'),
        pytest.param(
            False, [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]], False, id='False'
        ),
        pytest.param(
            kd.data.py.DropRemainder.DROP,
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            False,
            id='DROP',
        ),
        pytest.param(
            'drop',
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            False,
            id='drop',
        ),
        pytest.param(
            kd.data.py.DropRemainder.KEEP,
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
            False,
            id='KEEP',
        ),
        pytest.param(
            'keep',
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]],
            False,
            id='keep',
        ),
        pytest.param(
            kd.data.py.DropRemainder.PAD,
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 0]],
            False,
            id='PAD',
        ),
        pytest.param(
            'pad',
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 0]],
            False,
            id='pad',
        ),
        pytest.param(
            'keepp',
            [],
            True,
            id='Invalid',
        ),
    ],
)
def test_drop_remainder(
    batch_drop_remainder: bool | kd.data.py.DropRemainder,
    expected_output: list[list[int]],
    should_raise: bool,
):
  if should_raise:
    cm = pytest.raises(ValueError, match='is not a valid DropRemainder')
  else:
    cm = contextlib.nullcontext()
  with cm:
    ds = kd.data.py.DataSource(
        grain.RangeDataSource(0, 10, 1),
        shuffle=False,
        num_epochs=1,
        num_workers=0,
        batch_size=3,
        batch_drop_remainder=batch_drop_remainder,
    )
  if should_raise:
    return

  actual_output = list(ds)
  assert len(actual_output) == len(expected_output), (
      actual_output,
      expected_output,
  )
  for i, (actual, expected) in enumerate(zip(actual_output, expected_output)):
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=f'mismatch at index {i}: {actual=} vs {expected=}',
    )


def test_multi_host_pad_sharding():
  # Test with 16 hosts, batch_size=128 (host_batch_size=8), dataset length 140 (e.g. OVIS).
  # Padded length is ceil(140 / 128) * 128 = 256 items.
  # Each host should get 256 / 16 = 16 items -> exactly 2 batches of size 8.
  num_processes = 16
  global_batch_size = 128
  host_batch_size = 8
  ds_len = 140
  expected_num_batches = 2

  with mock.patch('jax.process_count', return_value=num_processes):
    for p_idx in range(num_processes):
      with mock.patch('jax.process_index', return_value=p_idx):
        ds = kd.data.py.DataSource(
            grain.RangeDataSource(0, ds_len, 1),
            shuffle=False,
            num_epochs=1,
            num_workers=0,
            batch_size=global_batch_size,
            batch_drop_remainder=kd.data.py.DropRemainder.PAD,
        )
        assert len(ds) == expected_num_batches
        batches = list(ds)
        assert len(batches) == expected_num_batches
        for b in batches:
          assert len(b) == host_batch_size


def test_multi_host_pad_sharding_error():
  num_processes = 2
  global_batch_size = 4
  ds_len = 10

  with mock.patch("jax.process_count", return_value=num_processes):
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch.object(
          grain.MapDataset,
          "repeat",
          side_effect=AttributeError("Dataset does not support repeat"),
      ):
        ds = kd.data.py.DataSource(
            grain.RangeDataSource(0, ds_len, 1),
            shuffle=False,
            num_epochs=1,
            num_workers=0,
            batch_size=global_batch_size,
            batch_drop_remainder=kd.data.py.DropRemainder.PAD,
        )
        with pytest.raises(
            AttributeError, match="Dataset does not support repeat"
        ):
          _ = ds.ds_for_current_process(jax.random.PRNGKey(0))





