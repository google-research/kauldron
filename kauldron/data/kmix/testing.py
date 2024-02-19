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

"""Test utils."""

from collections.abc import Iterator
import contextlib
import functools
from typing import Any
from unittest import mock

import grain.tensorflow as grain
import tensorflow as tf
import tensorflow_datasets as tfds


class _MockedDataSource(grain.TfDataSource):
  """Mocked data source."""

  def __init__(
      self,
      name: str,
      *,
      split: str,
      decoders: Any,
      num_examples: int,
      **kwargs,
  ):
    del kwargs
    with tfds.testing.mock_data(num_examples=1):
      ds = tfds.load(name, split=split, decoders=decoders)
    # TODO(epot): Why does grain uses `.unbatch()` internally ?
    (self.ex,) = tfds.as_numpy(ds.take(1))
    self.num_examples = num_examples

  def __len__(self) -> int:
    return self.num_examples

  def __getitem__(self, record_keys: tf.Tensor) -> Any:
    # TODO(b/325610230): Could be simplified if Grain changes its API.
    return tf.nest.map_structure(
        lambda x: tf.broadcast_to(
            tf.constant(x), tf.concat([tf.shape(record_keys), x.shape], axis=0)
        ),
        self.ex,
    )


@contextlib.contextmanager
def mock_data(num_examples: int = 1) -> Iterator[None]:
  with mock.patch.object(
      grain.TfdsDataSource,
      'from_name',
      functools.partial(_MockedDataSource, num_examples=num_examples),
  ):
    yield
