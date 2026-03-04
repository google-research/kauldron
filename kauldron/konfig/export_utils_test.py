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

"""Test functions for export utils."""

import dataclasses
import json
import typing
from typing import Any, NamedTuple
from etils import enp
from kauldron import konfig
import numpy as np


def test_array_spec_export_import():
  spec = {
      'img': enp.ArraySpec(shape=(64, 64, 3), dtype=np.uint8),
      'label': enp.ArraySpec(shape=(), dtype=np.float32),
  }
  spec_json = konfig.export(spec)
  spec_restored = konfig.resolve(spec_json, freeze=False)
  assert spec_restored == spec


@dataclasses.dataclass
class CustomSpec:
  data: np.ndarray
  dtype: Any = np.uint8


def test_array_export_import():
  spec = CustomSpec(data=np.zeros((10, 10)))
  spec_json = konfig.export(spec)
  spec_restored = konfig.resolve(spec_json, freeze=False)
  spec_restored = typing.cast(CustomSpec, spec_restored)
  assert spec_restored.data.shape == spec.data.shape
  assert spec_restored.dtype == spec.dtype
  assert spec_restored.data.dtype == spec.data.dtype


class A:
  """Custom class with a __konfig_export__ method."""

  def __init__(self, a=1):
    self._a = a

  def __konfig_export__(self):
    return {
        '__qualname__': konfig.export_qualname(self),
        'a': self._a,
    }


def test_konfig_export():
  a = A()
  a_json = konfig.export(a)
  a_restored = konfig.resolve(a_json, freeze=False)
  assert type(a_restored) == type(a) and a_restored.__dict__ == a.__dict__


def test_non_string_key():
  spec = {
      ('edge_a', ('src', 'dst')): np.array([1, 2, 3]),
      ('edge_b', ('x', 'y')): np.array([4, 5]),
  }
  spec_json = konfig.export(spec)
  json_str = json.dumps(spec_json, indent=2)
  spec_loaded = json.loads(json_str)
  spec_restored = konfig.resolve(spec_loaded, freeze=False)
  assert list(spec_restored.keys()) == list(spec.keys())
  for k in spec:
    np.testing.assert_array_equal(spec_restored[k], spec[k])


class MyKey(NamedTuple):
  foo: str
  bar: tuple[str, ...]


def test_non_string_key_dict_json_serializable():
  spec = {
      MyKey('x', ('a', 'b')): np.array([1, 2, 3]),
      MyKey('y', ('c', 'd')): np.array([4, 5]),
  }
  spec_json = konfig.export(spec)
  spec_restored = konfig.resolve(spec_json, freeze=False)
  assert isinstance(spec_restored, dict)
  assert type(list(spec_restored.keys())[0]) == type(list(spec.keys())[0])
