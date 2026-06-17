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

"""Knnx: a wrapper for nnx modules to be used in kauldron."""

# pylint: disable=g-importing-member, g-multiple-import

import copy
import dataclasses
import inspect
import typing as tp
from typing import Any, Callable, ClassVar, Optional
import warnings
import flax
from flax import linen as nn
from flax import nnx
from flax.core.scope import (
    CollectionFilter,
    DenyList,
)
from flax.typing import FrozenVariableDict
from kauldron import kd

if tp.TYPE_CHECKING:
  bases = (nn.Module, nnx.Module)
else:
  bases = (nnx.Module,)


@dataclasses.dataclass(kw_only=True)
class KdNnxModule(*bases):
  """A wrapper for torch modules to be used in kauldron.

  Inherits from nn.Module for typing purposes in kauldron.

  Example usage:
  ```python
  @dataclasses.dataclass(kw_only=True)
  class MyKdNnxModule(kd.contrib.knnx.KdNnxModule):
    input_dim: int = 3
    hdim: int = 10

    def setup(self, rngs: Optional[nnx.Rngs] = None):
      self.lin = nnx.Linear(self.input_dim, self.hdim, rngs=rngs)

    def __call__(self, x):
      return self.lin(x)
  ```

  Attributes:
    __dynamic_module__: Whether we allow the module structure to change during
      execution. If False, then the module is not re-split at every apply call
      to return the new module structure, which can save time.
  """

  __dynamic_module__: ClassVar[bool] = True

  def __init_subclass__(cls):
    assert hasattr(cls, 'setup'), """Subclass must implement setup method."""
    return super().__init_subclass__()

  def setup(self, rngs: Optional[nnx.Rngs] = None):
    raise NotImplementedError()

  def get_linen_variables(self, capture_intermediates=False):
    intermediates = {}
    if capture_intermediates:
      # poping intermediates before nnx.split
      intermediates = nnx.pop(self, nnx.Intermediate)._mapping  # pylint: disable=protected-access

    gdef, params, rng_vars, attributes = nnx.split(
        self, nnx.Param, nnx.Any(nnx.RngCount, nnx.RngKey), ...
    )
    return {
        'params': params._mapping,  # pylint: disable=protected-access
        'nnx': {
            'graphdef': gdef,
            'rng_vars': rng_vars._mapping,  # pylint: disable=protected-access
            'attributes': attributes._mapping,  # pylint: disable=protected-access
        },
        'intermediates': intermediates,
    }

  def init(
      self,
      rngs: (
          flax.typing.PRNGKey
          | flax.typing.RNGSequences
          | dict[str, kd.random.PRNGKey]
      ),
      *args: Any,
      method: Callable[..., Any] | str | None = None,
      mutable: CollectionFilter = DenyList('intermediates'),
      capture_intermediates: (
          bool | Callable[['KdNnxModule', str], bool]
      ) = False,
      **kwargs: Any,
  ) -> FrozenVariableDict | dict[str, Any]:
    """Initializes the NNX module and get variables.

    Args:
      rngs: The rngs to use for the module.
      *args: Positional arguments to pass to the module.
      method: The method to apply.
      mutable: Whether to return the modified collections.
      capture_intermediates: Whether to capture intermediates.
      **kwargs: Keyword arguments to pass to the module.

    Returns:
      The variables of the module.
    """

    # delete unused arguments
    del args, kwargs, method, capture_intermediates, mutable

    self_copy = copy.deepcopy(self)
    if 'rngs' in inspect.signature(self_copy.setup).parameters:
      self_copy.setup(rngs=_nnx_rngs_from_kd(rngs))
    else:
      warnings.warn(
          'rngs are not used in setup method. If your module requires rngs, '
          'please add them to the setup method.'
      )
      self_copy.setup()
    return self_copy.get_linen_variables()

  def apply(
      self,
      variables: flax.typing.VariableDict | dict[str, Any],
      *args: Any,
      rngs: (
          flax.typing.PRNGKey
          | flax.typing.RNGSequences
          | dict[str, kd.random.PRNGKey]
          | None
      ) = None,
      method: Callable[..., Any] | str | None = None,
      mutable: CollectionFilter = False,
      capture_intermediates: (
          bool | Callable[['KdNnxModule', str], bool]
      ) = False,
      is_training_property: bool = True,
      **kwargs: Any,
  ) -> Any | tuple[Any, FrozenVariableDict | dict[str, Any]]:
    """Applies the NNX module.

    Args:
      variables: The variables of the module.
      *args: Positional arguments to pass to the module.
      rngs: The rngs to use for the module.
      method: The method to apply.
      mutable: Whether to return the modified collections.
      capture_intermediates: Whether to capture intermediates.
      is_training_property: Whether to set the training mode.
      **kwargs: Keyword arguments to pass to the module.

    Returns:
      The output of the module, and optionally the modified collections.
    """
    if method is None:
      method = '__call__'
    # reseed rngs
    rng_vars = variables['nnx']['rng_vars']
    if rngs:
      new_rngs = _nnx_rngs_from_kd(rngs)
      rng_vars = _reseed_rng_vars(rng_vars, new_rngs)

    # re-create module from variables
    module = nnx.merge(
        variables['nnx']['graphdef'],
        variables['params'],
        rng_vars,
        variables['nnx']['attributes'],
    )

    # set training mode
    if is_training_property:
      module.train()
    else:
      module.eval()

    # get output
    if isinstance(method, str):
      method_fn = getattr(self.__class__, method)
    else:
      method_fn = method
    out = method_fn(module, *args, **kwargs)

    # return out and modified collections
    if mutable and self.__dynamic_module__:
      linen_vars = module.get_linen_variables(
          capture_intermediates=capture_intermediates
      )
      return out, linen_vars
    elif mutable:
      # mutate variables in place
      variables = {k: v for k, v in variables.items()}
      variables['nnx']['rng_vars'] = rng_vars
      variables['intermediates'] = nnx.pop(module, nnx.Intermediate)._mapping  # pylint: disable=protected-access
      return out, variables
    else:
      return out


@dataclasses.dataclass(kw_only=True)
class LinenModuleFromNnxDef(KdNnxModule):
  """A general wrapper for nnx modules.

  Attributes:
    nnx_init: The function to initialize the nnx module. it can be a class or a
      function that returns an nnx module. If you want to pass arguments to the
      nnx module, you can use a partial function.
  """

  nnx_init: Callable[..., nnx.Module]

  def setup(self, rngs: Optional[nnx.Rngs] = None):
    try:
      self.module = self.nnx_init(rngs=rngs)
    except TypeError:
      self.module = self.nnx_init()

  def __call__(self, *args, **kwargs):
    return self.module(*args, **kwargs)


def _nnx_rngs_from_kd(rngs: dict[str, kd.random.PRNGKey] | None = None):
  if rngs is None:
    return nnx.Rngs(0)
  if isinstance(next(iter(rngs.values())), kd.random.PRNGKey):
    rng_keys = {k: v.rng for k, v in rngs.items()}
  else:
    rng_keys = rngs
  return nnx.Rngs(**rng_keys)


T = tp.TypeVar('T')


def _reseed_rng_vars(rng_vars_state: T, rngs: nnx.Rngs) -> T:
  """Reseed the rng vars state with the given rngs.

  We use this on rng_vars coming from the split module, because nnx.reseed
  does not work on module when a nnx.Rngs object is stored at multiple places in
  the module.

  Args:
    rng_vars_state: The rng variables state from `nnx.split`.
    rngs: The new `nnx.Rngs` to reseed with.

  Returns:
    A deepcopy of `rng_vars_state` with reseeded `nnx.RngKey` values.
  """
  rng_vars_state = nnx.clone(rng_vars_state)
  for path, var in nnx.graph.iter_graph(rng_vars_state):
    if isinstance(var, nnx.RngKey):
      assert len(path) >= 2, 'Incorrect structure for input rng_vars_state.'
      # Only update variables that are in the rngs
      if var.tag in rngs:
        var.value = rngs[var.tag]()
  return rng_vars_state
