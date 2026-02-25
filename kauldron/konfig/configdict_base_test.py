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

import json
import pathlib
import types

from etils import epy
from kauldron import konfig
from kauldron.utils import immutabledict


def test_cycles():
  cfg = konfig.ConfigDict({
      'a0': 'abc',
      'c0': {
          'a': 123,
      },
  })
  cfg.a1 = cfg.get_ref('a0')
  cfg.c1 = cfg.get_ref('c0')
  cfg.cycle = cfg

  assert repr(cfg) == epy.dedent("""
    <ConfigDict[&id003 {
        'a0': &id001 'abc',
        'c0': &id002 {'a': 123},
        'a1': *id001,
        'c1': *id002,
        'cycle': *id003,
    }]>
  """)


def test_ref_resolved():
  cfg = konfig.ConfigDict({'a': 1})
  cfg.b = (1, cfg.ref.a)

  # Resolve should resolve the FieldReference
  out = konfig.resolve(cfg)
  assert isinstance(out.b[-1], int)
  assert hash(out)


def test_deserialize():
  cfg = konfig.ConfigDict({
      'paths': [
          {'__const__': 'pathlib:Path'},
          {'__const__': 'pathlib:Path'},
      ],
  })
  cfg = konfig.resolve(cfg)
  assert cfg['paths'] == (pathlib.Path, pathlib.Path)


def test_json_path():
  cfg = konfig.ConfigDict({
      'path': pathlib.Path('a/b'),
  })
  assert cfg.to_json() == '{"path": "a/b"}'


def test_dot():
  cfg = konfig.ConfigDict({
      'a.b.c': 123,
  })
  assert cfg['a.b.c'] == 123


def test_ref():
  cfg = konfig.ConfigDict()
  cfg.a = True
  cfg.b = konfig.ConfigDict({
      'b': {
          'a': cfg.ref.a,
      },
  })

  cfg.a = False
  assert cfg.b.b.a is False  # pylint: disable=g-bool-id-comparison


def test_aliases():
  repr_ = konfig.configdict_base._normalize_qualname
  assert repr_('tensorflow') == 'tf'
  assert repr_('tensorflow.gfile:exists') == 'tf.gfile.exists'
  assert repr_('tensorflow:int64') == 'tf.int64'
  assert repr_('tensorflow_graphics:Foo') == 'tensorflow_graphics.Foo'
  assert repr_('tensorflow_datasets:load') == 'tfds.load'


def test_expand_qualname():
  expand = konfig.configdict_base.expand_qualname
  assert expand('kd.data:Xxx') == 'kauldron.kd:data.Xxx'
  assert expand('kd:Trainer') == 'kauldron.kd:Trainer'
  assert expand('np:int32') == 'numpy:int32'
  assert expand('nn:Module') == 'flax.linen:Module'
  assert expand('my.module:Cls') == 'my.module:Cls'


def test_indices():
  data = {
      0: {
          '1': 1,
      },
      '0': 'aaa',
      'b': 2,
      -4: 'a',
  }
  cfg = konfig.ConfigDict(data)
  new_cfg = konfig.ConfigDict(json.loads(cfg.to_json()))
  assert cfg == new_cfg
  assert new_cfg[0]['1'] == 1
  assert new_cfg['0'] == 'aaa'
  assert 1 not in new_cfg[0]
  assert '1' in new_cfg[0]


def test_immutabledict():
  cfg = konfig.ConfigDict()
  cfg.a = immutabledict.ImmutableDict({'a': 1})

  assert cfg == konfig.ConfigDict({'a': {'a': 1}})


def test_module_configdict():
  cfg = konfig.module_configdict.AutoNestedConfigDict()
  cfg.a.a = 1
  cfg.a.b.c = [1, 2]
  cfg.a.b.d = {'e': 3}
  cfg.b = [{'a': [1, 3]}, [2, 3]]
  assert cfg.as_flat_dict() == {
      'a.a': 1,
      'a.b.c': [1, 2],
      'a.b.d': konfig.ConfigDict({'e': 3}),
      'b': [konfig.ConfigDict(dict({'a': [1, 3]})), [2, 3]],
  }
  cfg = konfig.module_configdict.AutoNestedConfigDict()
  cfg.a[0].b = dict(c=2)
  assert cfg.as_flat_dict() == {'a.0.b': konfig.ConfigDict(dict(c=2))}


def test_py_flag_overrides():
  cfg = konfig.ConfigDict({})

  overrides = {
      'x': 'py::types.SimpleNamespace(a=1)',
      'y': 'py::pathlib.Path',
      'y2': 'py::[pathlib.Path, 1]',
      'z': 'regular_string',
  }
  konfig.module_configdict._apply_overrides(cfg, overrides)

  assert cfg.x == konfig.ConfigDict({
      '__qualname__': 'types:SimpleNamespace',
      'a': 1,
  })
  assert cfg.y == konfig.ConfigDict({
      '__const__': 'pathlib:Path',
  })
  assert cfg.y2 == [
      konfig.ConfigDict({'__const__': 'pathlib:Path'}),
      1,
  ]
  assert cfg.z == 'regular_string'

  resolved = konfig.resolve(cfg, freeze=False)
  assert resolved == {
      'x': types.SimpleNamespace(a=1),
      'y': pathlib.Path,
      'y2': [pathlib.Path, 1],
      'z': 'regular_string',
  }
