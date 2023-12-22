# Copyright 2023 The kauldron Authors.
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

"""Ref utils test."""

import json

from etils import epy
from kauldron import konfig


def test_ref():
  cfg = konfig.ConfigDict(dict(a=10))
  cfg.greater_than_20 = cfg.ref.a + 15 > 20

  assert cfg.greater_than_20 is True  # pylint: disable=g-bool-id-comparison

  cfg.a = 0  # Updating a change `greater_than_20`
  assert cfg.greater_than_20 is False  # pylint: disable=g-bool-id-comparison

  cfg.schedules = {
      'learning_rate': 0.001,
  }
  cfg.learning_rate = cfg.ref.schedules['learning_rate']
  assert cfg.learning_rate == 0.001

  cfg.schedules['learning_rate'] = 10.0
  assert cfg.learning_rate == 10.0

  # TODO(epot): Test lazy values after `konfig.resolve`


def test_ref_fn():
  @konfig.ref_fn
  def _join_parts(parts: list[str]) -> str:
    return '/'.join(parts)

  cfg = konfig.ConfigDict()
  cfg.parts = ['a', 'b', 'c']
  cfg.joined_parts = _join_parts(cfg.ref.parts)
  cfg.joined_parts2 = konfig.ref_fn('/'.join, cfg.ref.parts)

  assert cfg.joined_parts == 'a/b/c'
  assert cfg.joined_parts2 == 'a/b/c'

  cfg.parts = ['d', 'e', 'f']
  assert cfg.joined_parts == 'd/e/f'
  assert cfg.joined_parts2 == 'd/e/f'


def test_ref_copy():
  train_ds = konfig.ConfigDict(
      dict(
          src=dict(
              name='mnist',
              version='1.0.0',
              split='train',
          ),
          shuffle=True,
          batch_size=32,
          prefetch=True,
      )
  )

  test_ds = konfig.ref_copy(train_ds)

  # Updating the copy won't update `train_ds`
  test_ds.src.split = 'test'
  test_ds.shuffle = False

  # Updating the original config update the ref_copy
  train_ds.src.version = '2.0.0'
  train_ds.batch_size = 128
  train_ds.shuffle = True

  assert json.loads(train_ds.to_json()) == dict(
      src=dict(
          name='mnist',
          version='2.0.0',
          split='train',
      ),
      shuffle=True,
      batch_size=128,
      prefetch=True,
  )
  assert json.loads(test_ds.to_json()) == dict(
      src=dict(
          name='mnist',
          version='2.0.0',
          split='test',
      ),
      shuffle=False,
      batch_size=128,
      prefetch=True,
  )

  # Caveat

  # New attribute
  train_ds.new_attribute = 123
  assert not hasattr(test_ds, 'new_attribute')

  # Overwritte dict
  train_ds.src = dict(
      name='imagenet',
      split='train',
  )
  train_ds.src.split = 'eval'
  assert train_ds.src.name == 'imagenet'
  assert train_ds.src.split == 'eval'
  assert test_ds.src.name == 'mnist'
  assert test_ds.src.split == 'test'


def test_ref_future_shared_val():
  cfg = konfig.ConfigDict()

  cfg.model = {
      'encoder': {'a': 123},
      'decoder': {'a': cfg.ref.model.encoder},
  }
  # TODO(epot): Fix: `&id001` should appear
  assert repr(cfg) == epy.dedent("""
  <ConfigDict[{
      'model': {
          'decoder': {'a': None},
          'encoder': *id001,
      },
  }]>
  """)
  # TODO(epot): Serialization/deserialization
  # TODO(epot): Resolve


def test_ref_repr():
  cfg = konfig.ConfigDict({
      'a': 123,
  })
  cfg.b = cfg.ref.a * 2
  assert repr(cfg) == epy.dedent("""
  <ConfigDict[{
      'a': 123,
      'b': 246,
  }]>
  """)
  # TODO(epot): Serialization
  # TODO(epot): Resolve


def test_ref_repr_future_error():
  cfg = konfig.ConfigDict({})
  cfg.a = {'a': cfg.ref.non_existing * 2}
  assert repr(cfg) == """
  <ConfigDict[{'a': {'a': <Unresolved>}}]>
  """
  # TODO(epot): Serialization
  # TODO(epot): Resolve should trigger a very good error message
