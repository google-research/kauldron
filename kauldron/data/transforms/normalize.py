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

"""Normalize utils."""

from __future__ import annotations

from collections.abc import Callable
import types
import typing
from typing import Any

from etils import epy
from kauldron.data.transforms import abc as tr_abc

if typing.TYPE_CHECKING:
  _grain = Any  # Either TfGrain or PyGrain  # pylint: disable=invalid-name
  _Transformation = tr_abc.Transformation | _grain.Transformation


def normalize_transform(
    tr: _Transformation,
    *,
    grain_module: types.ModuleType,  # Literal[_grain]
    kd_to_grain_transform: dict[
        type[tr_abc.Transformation], type[_grain.Transformation]
    ],
    grain_transform_to_apply_wrapper: (
        None
        | dict[
            type[_grain.Transformation],
            Callable[[type[_grain.Transformation]], None],
        ]
    ) = None,
) -> _grain.Transformation:
  """Convert the kd transform to PyGrain or TfGrain."""
  if isinstance(tr, grain_module.Transformation):
    return tr
  if not isinstance(tr, tr_abc.Transformation):
    raise TypeError(
        f'Unexpected transform: {type(tr).__qualname__} is neither a'
        ' `grain.Transformation` nor a Kauldron transform object. Are you'
        ' mixing PyGrain and TfGrain ?'
    )

  # Find the grain classe to inherit from
  for kd_cls, grain_cls in kd_to_grain_transform.items():
    if isinstance(tr, kd_cls):
      break
  else:
    raise TypeError(
        f'Unexpected Kauldron transform: {type(tr).__qualname__}. This is'
        ' likely a Kauldron bug. Please repoort.'
    )

  tr_cls = type(tr)

  @epy.wraps_cls(tr_cls)
  class _WrappedTransform(tr_cls, grain_cls):
    """Wrapped transform."""

    def __init__(self):
      # Transform should be immutable, so should be fine
      self.__dict__.update(tr.__dict__)

  if grain_transform_to_apply_wrapper:
    # Eventually filter the `grain.META_FEATURES`
    grain_transform_to_apply_wrapper[grain_cls](_WrappedTransform)

  wrapped_tr = _WrappedTransform()
  return wrapped_tr
