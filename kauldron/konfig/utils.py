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
import os
import typing
from typing import Annotated, Any, TypeVar

from etils import epath
import ml_collections

_T = TypeVar('_T')
# No-op annotation to indicate the object should be a ConfigDict object:
# * `x: MyObj`: Resolved object
# * `x: ConfigDictLike[MyObj]`: ConfigDict object, but allow auto-complete
if not typing.TYPE_CHECKING:
  ConfigDictLike = Annotated[_T, None]
else:
  # TODO(b/254514368): Remove hack to make the alias work with PyType

  class _ConfigDictLikeMeta(type):

    def __getitem__(cls, obj):
      return obj

  class ConfigDictLike(metaclass=_ConfigDictLikeMeta):
    pass


# Wrapper around `placeholder` which accept any default
@functools.wraps(ml_collections.config_dict.placeholder)
def placeholder(field_type: Any = object, *, required: bool = False) -> Any:
  """Defines an entry in a ConfigDict that has no value yet."""
  return ml_collections.config_dict.placeholder(field_type, required=required)


class DefaultJSONEncoder(ml_collections.config_dict.CustomJSONEncoder):
  """Default JSONEncoder."""

  def default(self, obj):
    if isinstance(obj, epath.PathLikeCls):
      return os.fspath(obj)
    # Could add protocol support to allow encode arbitrary json objects
    return super().default(obj)
