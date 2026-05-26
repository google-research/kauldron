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

"""Deprecated backward-compatibility alias for immutabledict."""

import warnings
import flax.core
from typing import Any

warnings.warn(
    '`kauldron.konfig.immutabledict_lib` is deprecated and will be removed. '
    'Please migrate to standard dict or `flax.core.FrozenDict`.',
    DeprecationWarning,
    stacklevel=2,
)

ImmutableDict = flax.core.FrozenDict


def freeze_dict_attrs(obj: Any, attrs: list[str]) -> None:
  """Deprecated fallback for freeze_dict_attrs."""
  warnings.warn(
      '`freeze_dict_attrs` is deprecated. Default behavior is no freeze.',
      DeprecationWarning,
      stacklevel=2,
  )
  for attr in attrs:
    val = getattr(obj, attr)
    if isinstance(val, dict):
      setattr(obj, attr, flax.core.FrozenDict(val))


def freeze(x: Any) -> Any:
  """Deprecated fallback for freeze."""
  return flax.core.FrozenDict(x) if isinstance(x, dict) else x
