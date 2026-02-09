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

"""Utilities for exporting python objects to json-serializable dicts."""

import dataclasses
import inspect
from typing import Any

from etils import enp
from etils import epy
from kauldron.utils import immutabledict


def export_qualname(obj: Any) -> str:
  """Returns the qualified name of the object."""
  return f'{type(obj).__module__}:{type(obj).__name__}'


def export(obj: Any) -> epy.typing.Json:
  """Exports obj to a json-serializable dict.

  Example usage:
  ```
  data = konfig.export(obj)
  ```

  Json can be serialized/deserialized (e.g. to save on disk)
  ```
  data = json.loads(json.dumps(data))
  ```

  The dict can be resolved with konfig to get the original object back
  ```
  assert konfig.resolve(data, freeze=False) == obj
  ```

  Some types are natively supported: list, tuple, dict, numpy arrays,
  jax arrays, immutabledict, numpy dtypes, and dataclasses.

  Custom classes can implement a `__konfig_export__` method to be serializable.
  The method should return a dict with a `__qualname__` key and exportable
  values, e.g.:
  ```
  def __konfig_export__(self):
    return {
        '__qualname__': konfig.export_qualname(self),
        'a': self.a,
        'b': self.b,
    }
  ```

  Args:
    obj: The object to export.

  Returns:
    The exported object.
  """
  if inspect.getattr_static(obj, '__konfig_export__', None) is not None:
    exported = obj.__konfig_export__()
    if not isinstance(exported, dict) or '__qualname__' not in exported:
      raise ValueError(
          f'Wrong value for object {obj} exported with `__konfig_export__`.'
          f'Expected a dict with a "__qualname__" key, got {exported} from'
          f' {obj.__class__.__name__}. Please explicitly add'
          f' "__qualname__": "{export_qualname(obj)}" to the returned'
          ' dict.'
      )
    # otherwise, we export the dict values.
    fields = {k: export(v) for k, v in exported.items()}
  elif dataclasses.is_dataclass(obj):
    fields = {
        f.name: export(getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if f.init
    }
    fields['__qualname__'] = export_qualname(obj)
  elif enp.lazy.is_np(obj):
    fields = {
        '__qualname__': 'numpy:asarray',
        '0': obj.tolist(),
        'dtype': obj.dtype.name,
    }
  elif enp.lazy.is_jax(obj):
    fields = {
        '__qualname__': 'jax.numpy:asarray',
        '0': obj.tolist(),
        'dtype': obj.dtype.name,
    }
  elif isinstance(obj, immutabledict.ImmutableDict):
    fields = {k: export(v) for k, v in obj.items()}
    fields['__qualname__'] = export_qualname(obj)
  elif isinstance(obj, enp.ArraySpec):
    # TODO(geco): In theory this is not needed as ArraySpec now defines its own
    # __konfig_export__. However, there are binaries that have been built before
    # this was added, so we keep this option for backward compatibility.
    fields = {
        '__qualname__': export_qualname(obj),
        'shape': obj.shape,
        'dtype': obj.dtype.name,
    }
  elif isinstance(obj, dict):
    fields = {k: export(v) for k, v in obj.items()}
    if type(obj) != dict:  # pylint: disable=unidiomatic-typecheck
      # if it is a subclass of dict, we can probably re-create the object
      # as follows:
      fields['__qualname__'] = export_qualname(obj)
  elif isinstance(obj, list):
    fields = [export(v) for v in obj]
  elif isinstance(obj, tuple):
    fields = {'__qualname__': 'builtins.tuple', '0': [export(t) for t in obj]}
  elif isinstance(obj, int | float | type(None) | str | bool):
    # here we directly return the value since there is no sub-call.
    return obj
  elif 'numpy.dtype' in str(obj.__class__):
    # check for numpy dtype without importing numpy.
    return {'__qualname__': 'numpy:dtype', '0': obj.name}
  elif isinstance(obj, type):
    return {'__const__': f'{obj.__module__}:{obj.__name__}'}
  else:
    raise NotImplementedError(
        f'Unable to export object {obj}: Unknown object type{obj.__class__}.'
        ' Add relevant case in `konfig.configdict_proxy.export` or define a'
        f' custom __konfig_export__ method in {obj.__class__}.'
    )

  return fields
