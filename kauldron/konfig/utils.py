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

"""Utils."""

import functools
import json
import os
import typing
from typing import Any, TypeVar

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
    required: bool = False,
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


class DefaultJSONEncoder(json.JSONEncoder):
  """Default JSONEncoder."""

  def default(self, o):
    obj = o  # base class name is too short
    match obj:
      case ml_collections.FieldReference():
        # TODO(epot): Support field-ref (at least with identity), so
        # deserialization restore the ref.
        return obj.get()
      case ml_collections.ConfigDict():
        return {k: v for k, v in obj._fields.items()}
      case os.PathLike():  # For convenience, allow `pathlib`-like objects
        return os.fspath(obj)
      case _:
        raise TypeError(
            '{} is not JSON serializable. Instead use '
            'ConfigDict.to_json_best_effort()'.format(type(obj))
        )
