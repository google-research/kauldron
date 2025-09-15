# Copyright 2025 The kauldron Authors.
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

"""Module to Wrap a Nnx module into a linen module."""

# pylint: disable=g-importing-member
# pylint: disable=g-multiple-import
# pylint: disable=protected-access

from functools import partial
import inspect
from typing import Any, Callable, Mapping, Sequence
import flax
from flax import linen as nn
from flax import nnx
import jax
from kauldron import kd


class LinenFromNnxDef(nn.Module):
  """Wrap a nnx module into a linen module.

    This delays instantiation of the nnx and only instantiate it at `init` phase
    of the linen module. Then it splits the nnx module into different components
    with nnx.split, saves the graphdef, params, and other variables into scope.

    At `apply` phase, it fetches the elements from the scope, then merges them
    together with `nnx.merge` to recover the nnx module and make a forward pass.

    Usage:
    `linen_module = linen_from_nnx(nnx_class, *args, **kwargs)`
    e.g.
    ```
    model = linen_from_nnx(
      nnx.Linear,
      in_features=3,
      out_features=4,
  )
  ```
  Attributes:
    nnx_class: The nnx class to wrap.
    args: Positional arguments to pass to the nnx class `__init__`.
    kwargs: Keyword arguments to pass to the nnx class `__init__`.
  """

  nnx_class: Callable[..., nnx.Module]
  args: Sequence = ()  # pylint: disable=g-bare-generic
  kwargs: Mapping[str, Any] = flax.core.FrozenDict({})

  is_training = kd.nn.train_property()

  def instantiate_nnx_module(
      self, rngs: dict[str, Any] | None = None
  ) -> nnx.Module:
    """Initializes the NNX module and returns it.

    It recursively instantiates submodules that are part or args or kwargs, and
    that are also of type LinenFromNnxDef. This is needed to define
    hierarchical modules in konfig without instantiating them at during
    `konfig.resolve()`.

    If this is called inside a bound module, the rngs need to match the
    If this is not called inside a bound module, it can still work but requires
    to pass the rngs needed by the nnx module as a dict {rng_key_name: rng_key}.

    Args:
      rngs: Optional rng keys. If not set, the model will try to get them from
        the scope. The rngs keys will be fed to the nnx module if it requires
        them for init, and will be used to initialize submodules.

    Returns:
      The NNX module.
    """
    module_kwargs = dict(self.kwargs)
    module_args = self.args
    if rngs is None:
      rngs = _nnx_rngs_from_scope(self.scope)

    # nnx rngs are stateful so we can re-use them to instantiate sub-modules
    def _as_nnx(module: nnx.Module | LinenFromNnxDef) -> nnx.Module:
      if isinstance(module, LinenFromNnxDef):
        return module.instantiate_nnx_module(rngs)
      return module  # Do not traverse the `nnx.Module`

    module_args = jax.tree.map(
        _as_nnx,
        module_args,
        is_leaf=lambda x: isinstance(x, (LinenFromNnxDef, nnx.Module)),
    )
    module_kwargs = jax.tree.map(
        _as_nnx,
        module_kwargs,
        is_leaf=lambda x: isinstance(x, (LinenFromNnxDef, nnx.Module)),
    )

    if 'rngs' in inspect.signature(self.nnx_class.__init__).parameters:
      # some nnx modules require rngs for init, in which case we should provide
      # them
      module_kwargs |= dict(rngs=rngs)

    module = self.nnx_class(*module_args, **module_kwargs)
    return module

  @property
  def _nnx_module(self) -> nnx.Module:
    """Get NNX module from its element in the scope."""
    if self.is_initializing():
      return self.instantiate_nnx_module()

    module = nnx.merge(
        self.get_variable('nnx', 'graphdef'),
        self.variables['params'],
        self.get_variable('nnx', 'rng_vars'),
        self.get_variable('nnx', 'attributes'),
    )

    if self.is_training:
      module.train()
    else:
      module.eval()

    return module

  def __getattr__(self, attr: str) -> Any:
    """Allows to call methods of the nnx module from the linen wrapper."""
    if hasattr(self.nnx_class, attr):
      return partial(self.__call__, method=attr)
    else:
      try:
        return getattr(self._nnx_module, attr)
      except AttributeError as exc:
        raise AttributeError(
            f'Wrapped NNX module has not attribute {attr}'
        ) from exc

  @nn.compact
  def __call__(self, *args, method: str = '__call__', **kwargs) -> Any:
    """Makes a forward with the NNX module.

    The NNX graphdef, rng keys and static attributes are saved in scope.
    Params are saved to scope only during initialization.

    Args:
      *args: Positional arguments to pass to the NNX module.
      method: The method to call on the NNX module.
      **kwargs: Keyword arguments to pass to the NNX module.
    """

    module = self._nnx_module
    outputs = getattr(module, method)(*args, **kwargs)

    # update variables in scope
    gdef, params, rng_vars, attributes = nnx.split(
        module, nnx.Param, nnx.Any(nnx.RngCount, nnx.RngKey), ...
    )
    # Save the graph def and other variables.
    if self.is_mutable_collection('nnx'):
      self.put_variable('nnx', 'graphdef', gdef)
      self.put_variable('nnx', 'rng_vars', rng_vars._mapping)
      self.put_variable('nnx', 'attributes', attributes._mapping)

    if self.is_initializing():
      for k, v in params._mapping.items():
        self.put_variable('params', k, v)

    return outputs


def linen_from_nnx(
    cls,
    *args,
    **kwargs,
):
  """Main entry point for converting a nnx class to a linen module."""

  return LinenFromNnxDef(
      nnx_class=cls, args=args, kwargs=flax.core.FrozenDict(kwargs)
  )


def _nnx_rngs_from_scope(scope: flax.core.scope.Scope) -> nnx.Rngs:
  """Get the nnx rngs from the scope.

  If using kauldron, we need to extract two consecutive wrappers to get the rng
  key values: a LazyRng wrapper and the kd_random.PRNGKey wrapper

  Args:
    scope: scope to extract rng keys from

  Returns:
    rng keys in the nnx format.
  """
  rng_keys = scope.rngs
  if isinstance(next(iter(rng_keys.values())), flax.core.scope.LazyRng):
    rng_keys = {k: v.rng for k, v in rng_keys.items()}
  # if there is a lazy rng wrapper, probably there is also a kauldron PRNGKey
  # wrapper that we need to remove.
  # so the following *if* should not be an *elif*.
  if isinstance(next(iter(rng_keys.values())), kd.random.PRNGKey):
    rng_keys = {k: v.rng for k, v in rng_keys.items()}
  return nnx.Rngs(**rng_keys)
