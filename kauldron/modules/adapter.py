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

"""Adapter layers."""

from typing import Any, Optional

from flax import linen as nn
from kauldron.utils import train_property


class WrapperModule(nn.Module):
  """Base class to wrapper a module.

  The wrapper module transparent with respect to the inner parameters (
  `{'params': inner_params}` instead of nesting
  `{'params': {'model': inner_params}}`).
  """

  model: nn.Module

  def __post_init__(self):
    super().__post_init__()
    # Share scope, to make the wrapper module transparent with respect to the
    # parameters (instead of nesting `{'params': {'model': params}}`).
    if self.scope is not None:
      nn.share_scope(self, self.model)


class ExternalModule(WrapperModule):
  """Module that is defined outside Kauldron.

  This is a **very** thin wrapper around `flax.linen.Module` that add:

  * Keys: To connect the model to the dataset batch
  * Training property compatibility: Pass `train=True` (or equivalent kwargs)
    when calling the model, rather than using the `kd.nn.train_property()`

  ```
  cfg.model = kd.nn.ExternalModule(
      model=nn.Dropout(),
      keys={
          'x': 'batch.image',
      },
      train_kwarg_name='~deterministic',
  )
  ```

  Attributes:
    model: The flax model to wrap
    keys: Mapping from `model.__call__` kwargs names to context paths ( e.g.
      `keys={'x': 'batch.image'}` to call the model as `model.apply(rng,
      x=batch['image'])`). If `str` given, the input is passed as `args`, if
      `dict`, the inputs are passed as `kwargs`.
    train_kwarg_name: If provided, then the model will be called with
      `model.apply(..., <train_kwarg_name>=True)`. Flax models don't have a
      standard way to specify train/eval mode, so each codebase uses a different
      convention (`deterministic=`, `train=`, `is_training=`,...). The kwargs
      can be inverted with `~` (e.g. `train_kwarg_name='~deterministic'`)
  """

  keys: str | dict[str, str]
  train_kwarg_name: Optional[str] = None

  is_training = train_property.train_property()

  @nn.compact
  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if '__args__' in kwargs:
      args = (kwargs.pop('__args__'),)

    train_kwarg = {}
    if self.train_kwarg_name is not None:
      is_training = self.is_training
      train_kwarg_name = self.train_kwarg_name
      if train_kwarg_name.startswith('~'):  # Invert `~deterministic`
        train_kwarg_name = train_kwarg_name.removeprefix('~')
        is_training = not is_training
      train_kwarg[train_kwarg_name] = is_training

    return self.model(*args, **kwargs, **train_kwarg)

  def __kontext_keys__(self) -> dict[str, str]:
    """Kauldron keys when calling `kontext.get_from_keys_obj`."""
    if isinstance(self.keys, str):
      return {'__args__': self.keys}
    else:
      return self.keys
