# Copyright 2025 The kauldron Authors.
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

from etils import enp
from kauldron import kd
import numpy as np
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
