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
from etils import enp
from grain import python as grain
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


def test_multi_host_pad_wraps_in_padded_element():
  """Tests that multi-host padding wraps elements in PaddedElement."""
  from unittest import mock
  import jax

  # 10 elements across 4 hosts, batch_size=4 (host_batch_size=1).
  # total_batch_size = 4, num_batches = ceil(10/4) = 3, padded_len = 12.
  # Each host receives 12 // 4 = 3 elements.
  all_host_elements = []
  for proc_idx in range(4):
    with (
        mock.patch.object(jax, 'process_count', return_value=4),
        mock.patch.object(jax, 'process_index', return_value=proc_idx),
    ):
      ds = kd.data.py.DataSource(
          grain.RangeDataSource(0, 10, 1),
          shuffle=False,
          num_epochs=1,
          num_workers=0,
          batch_size=4,
          batch_drop_remainder=kd.data.py.DropRemainder.PAD,
      )
      raw_ds = ds.ds_for_current_process(rng=jax.random.PRNGKey(0))
      elements = list(raw_ds)
      all_host_elements.append(elements)

  # Verify every host has exactly 3 elements
  for proc_idx, elements in enumerate(all_host_elements):
    assert len(elements) == 3, f'Host {proc_idx} received {len(elements)} items'

  # Host 0 (indices 0, 4, 8) -> all real
  assert all_host_elements[0] == [
      kd.data.py.PaddedElement(element=0, is_padding=False),
      kd.data.py.PaddedElement(element=4, is_padding=False),
      kd.data.py.PaddedElement(element=8, is_padding=False),
  ]
  # Host 1 (indices 1, 5, 9) -> all real
  assert all_host_elements[1] == [
      kd.data.py.PaddedElement(element=1, is_padding=False),
      kd.data.py.PaddedElement(element=5, is_padding=False),
      kd.data.py.PaddedElement(element=9, is_padding=False),
  ]
  # Host 2 (indices 2, 6, 10) -> 2 real, 1 padding
  assert all_host_elements[2] == [
      kd.data.py.PaddedElement(element=2, is_padding=False),
      kd.data.py.PaddedElement(element=6, is_padding=False),
      kd.data.py.PaddedElement(element=None, is_padding=True),
  ]
  # Host 3 (indices 3, 7, 11) -> 2 real, 1 padding
  assert all_host_elements[3] == [
      kd.data.py.PaddedElement(element=3, is_padding=False),
      kd.data.py.PaddedElement(element=7, is_padding=False),
      kd.data.py.PaddedElement(element=None, is_padding=True),
  ]


def test_multi_host_pad_exact_multiple_no_wrapping():
  """Tests that when dataset divides evenly, elements remain raw."""
  from unittest import mock
  import jax

  # 12 elements across 4 hosts, batch_size=4 -> exact multiple (12 == 12)
  for proc_idx in range(4):
    with (
        mock.patch.object(jax, 'process_count', return_value=4),
        mock.patch.object(jax, 'process_index', return_value=proc_idx),
    ):
      ds = kd.data.py.DataSource(
          grain.RangeDataSource(0, 12, 1),
          shuffle=False,
          num_epochs=1,
          num_workers=0,
          batch_size=4,
          batch_drop_remainder=kd.data.py.DropRemainder.PAD,
      )
      raw_ds = ds.ds_for_current_process(rng=jax.random.PRNGKey(0))
      elements = list(raw_ds)
      assert len(elements) == 3
      # Elements are raw integers, not PaddedElement instances
      for elem in elements:
        assert isinstance(elem, int)

