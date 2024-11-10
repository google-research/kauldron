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

"""Utils."""

import ast
import dataclasses
import functools
import itertools
import os
import typing
from typing import Any, Generic, TypeVar

import ml_collections

_T = TypeVar('_T')
# No-op annotation to indicate the object should be a ConfigDict object:
# * `x: MyObj`: Resolved object
# * `x: ConfigDictLike[MyObj]`: ConfigDict object, but allow auto-complete
if not typing.TYPE_CHECKING:
  ConfigDictLike = typing.Annotated[_T, None]
else:
  # TODO(b/254514368): Remove hack to make the alias work with PyType

  class _ConfigDictLikeMeta(type):

    def __getitem__(cls, obj):
      return obj

  class ConfigDictLike(metaclass=_ConfigDictLikeMeta):
    pass


# Wrapper around `placeholder` which accept any default
# Returns `Any` as it can be assigned to every attribute
@functools.wraps(ml_collections.config_dict.placeholder)
def placeholder(
    field_type: Any = object,
    *,
    required: bool = False,  # pylint: disable=redefined-outer-name
    default: Any = None,
) -> Any:
  """Defines an entry in a ConfigDict that has no value yet.

  Args:
    field_type: type of value.
    required: If `True`, the placeholder will raise an error on access if the
      underlying value hasn't been set.
    default: Can specify the default value.

  Returns:
    A `FieldReference` with value None and the given type.
  """
  return ml_collections.FieldReference(
      default, field_type=field_type, required=required
  )


# TODO(epot): konfig should raise error if `_required` field is not set. This
# could be done by having resolve check for the presence of any placeholder
# not set.
def required(field_type: type[Any]) -> Any:
  """Defines a required attribute in the config that has no value yet."""
  return placeholder(field_type)


@typing.runtime_checkable
class ConfigDictConvertible(typing.Protocol):
  """Protocol to convert a Python object into it's `konfig.ConfigDict`."""

  def __as_konfig__(self) -> Any:
    raise NotImplementedError()


def to_json(obj: Any) -> str:
  """Convert a ConfigDict to JSON."""
  return _ToJson().convert(obj)


@dataclasses.dataclass(slots=True, kw_only=True)
class CachedObj(Generic[_T]):
  """Keep track of the config with its associated resolved value.

  Keeps a reference to the config object so that Python's garbage collector
  doesn't clear the object from memory. As otherwise its `id(config)` could be
  taken by another unrelated config2 object. This can be the case with ref_fn
  where the config object is dynamically created every time (thus is not kept
  alive).
  """

  ref: Any
  value: _T


class _ToJson:
  """Convert a ConfigDict to JSON."""

  def __init__(self):
    self._id_to_json: dict[int, CachedObj[Any]] = {}
    self._counter = itertools.count()

  def convert(self, obj: Any) -> Any:
    """Convert a ConfigDict to JSON nested dict."""
    # Object already encountered: Cycles & support shared references
    if id(obj) in self._id_to_json:
      json_ = self._id_to_json[id(obj)].value
      # Eventually assign the `__id__` so deserializing conserve the shared
      # objects.
      if isinstance(json_, dict) and '__id__' not in json_:
        json_['__id__'] = next(self._counter)
      return json_

    # New object: Convert it to json
    json_ = self._convert_inner(obj)
    self._id_to_json[id(obj)] = CachedObj(ref=obj, value=json_)
    return json_

  def _convert_inner(self, obj: Any) -> Any:
    """Convert an object to JSON."""

    match obj:
      case ml_collections.ConfigDict():
        return {_encode_key(k): self.convert(v) for k, v in obj._fields.items()}
      case dict():
        return {_encode_key(k): self.convert(v) for k, v in obj.items()}
      case ml_collections.FieldReference():
        # TODO(epot): Support field-ref (at least with identity), so
        # deserialization restore the ref.
        return self.convert(obj.get())
      case list() | tuple():
        return [self.convert(v) for v in obj]
      case os.PathLike():  # For convenience, allow `pathlib`-like objects
        return os.fspath(obj)
      case int() | float() | bool() | str() | None:
        return obj
      case _:
        raise TypeError(f'{type(obj)} is not JSON serializable.')


# Json exports only support `str` keys. To allow restoring json configs,
# non-str keys are encoded/decoded to str.
_JSON_RAW_PREFIX = '__raw__='


def _encode_key(key: Any) -> str:
  """Encode a key to a JSON-compatible string."""
  if isinstance(key, str):
    return key
  elif isinstance(key, (int, bool, float)):
    return f'{_JSON_RAW_PREFIX}{key!r}'
  else:
    raise TypeError(
        f'Key {key!r} is not (yet) JSON serializable. Please open an issue if'
        ' you need this.'
    )


def maybe_decode_json_key(key: Any) -> Any:
  """Decode a key from a JSON-compatible string."""
  if isinstance(key, str) and key.startswith(_JSON_RAW_PREFIX):
    return ast.literal_eval(key.removeprefix(_JSON_RAW_PREFIX))
  else:
    return key
