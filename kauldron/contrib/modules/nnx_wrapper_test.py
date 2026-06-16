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

from __future__ import annotations

from flax import nnx
from kauldron import kontext
from kauldron.contrib.modules import nnx_wrapper


class NnxWithKeys(nnx.Module):
  image: kontext.Key = 'batch.image'
  label: kontext.Key = 'batch.label'

  def __init__(self, rngs):
    self.linear = nnx.Linear(3, 4, rngs=rngs)

  def __call__(self, image, label):
    del label
    return self.linear(image)


class NnxWithoutKeys(nnx.Module):

  def __init__(self, rngs):
    self.linear = nnx.Linear(3, 4, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)


class NnxWithOptionalKey(nnx.Module):
  image: kontext.Key = 'batch.image'
  mask: None | kontext.Key = None

  def __init__(self, rngs):
    self.linear = nnx.Linear(3, 4, rngs=rngs)

  def __call__(self, image, mask=None):
    del mask
    return self.linear(image)


def test_is_key_annotated_with_keys():
  model = nnx_wrapper.linen_from_nnx(NnxWithKeys)
  assert kontext.is_key_annotated(model)


def test_get_keypaths_without_keys():
  model = nnx_wrapper.linen_from_nnx(NnxWithoutKeys)
  assert not kontext.get_keypaths(model)


def test_get_keypaths_class_defaults():
  model = nnx_wrapper.linen_from_nnx(NnxWithKeys)
  keypaths = kontext.get_keypaths(model)
  assert keypaths == {'image': 'batch.image', 'label': 'batch.label'}


def test_get_keypaths_kwargs_override():
  model = nnx_wrapper.linen_from_nnx(NnxWithKeys, image='batch.rgb')
  keypaths = kontext.get_keypaths(model)
  assert keypaths == {'image': 'batch.rgb', 'label': 'batch.label'}


def test_get_keypaths_optional_key():
  model = nnx_wrapper.linen_from_nnx(NnxWithOptionalKey)
  keypaths = kontext.get_keypaths(model)
  assert keypaths == {'image': 'batch.image', 'mask': None}


def test_resolve_from_keyed_obj():
  model = nnx_wrapper.linen_from_nnx(NnxWithKeys)
  context = {'batch': {'image': [1, 2, 3], 'label': [0, 1, 0]}}
  result = kontext.resolve_from_keyed_obj(context, model)
  assert result == {'image': [1, 2, 3], 'label': [0, 1, 0]}
