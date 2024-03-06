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

"""Base Module class."""

from __future__ import annotations

from collections.abc import Callable
import contextlib
import dataclasses
import functools
import typing
from typing import Any, Iterator, Optional, TypeVar

from etils import edc
from etils import enp
from etils.etree import jax as etree
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from kauldron import random
from kauldron.klinen import intermediate
from kauldron.klinen import intermediate_proxy
from kauldron.klinen import traverse
from kauldron.klinen.collections import Collection
import numpy as np

_FnT = TypeVar('_FnT', bound=Callable)
_SelfT = TypeVar('_SelfT')


def _bind_only_method(fn: _FnT) -> _FnT:
  """Validate the method is only called after `init_bind()`."""

  @functools.wraps(fn)
  def new_fn(self: Module, *args, **kwargs):
    if not self._is_bind:  # pylint: disable=protected-access
      raise ValueError(
          f'Cannot call {fn.__qualname__} before calling .init_bind()'
      )
    return fn(self, *args, **kwargs)

  return new_fn


def _skip_wrap_call(fn: _FnT) -> _FnT:
  """Skip the flax function auto-wrapping."""
  # flax wrap all method inside `_call_wrapped_method`. Do not wrap
  # `_call_wrapped_method` to avoid infinite recursion
  fn.method_handler_wrapped = True
  return fn


@edc.dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class _ModuleState:
  """Module state.

  Only the root module has a state. The childs modules only uses the
  root state.

  Attributes:
    params: Bounded variables
    streams: Rng stream names
    rngs: Current rngs
    training: Whether model is in training or evaluation mode.
    tree_params_only: Whether only the tree are mapped over
  """

  params: Optional[flax.core.scope.FrozenVariableDict]
  streams: tuple[str, ...]
  rngs: dict[str, random.PRNGKey]
  training: bool = True
  tree_params_only: bool = False

  def replace(self: _SelfT, **kwargs: Any) -> _SelfT:
    return dataclasses.replace(self, **kwargs)


@edc.dataclass
@dataclasses.dataclass
class _Context:
  """Global context.

  Attributes:
    in_call_state: `_ModuleState` of the top level `y = model(x)` call.
    capture_proxy: If set, the intermediate values are forwarded to this proxy.
  """

  in_call_state: edc.ContextVar[Optional[_ModuleState]] = None
  capture_proxy: edc.ContextVar[
      Optional[intermediate_proxy.ModuleIntermediateProxy]
  ] = None

  @contextlib.contextmanager
  def set_in_call_state(self, module: Module) -> Iterator[None]:
    self.in_call_state = module._kd_state  # pylint: disable=protected-access
    try:
      yield
    finally:
      self.in_call_state = None


context = _Context()


class Module(nn.Module):  # pytype: disable=invalid-function-definition
  """Base Module class."""

  _: dataclasses.KW_ONLY  # Required to allow sub-classing
  # TODO(epot): Should be hidden from the public API
  # Fields for auto-complete/type-checking, but ignored by `@dataclass`
  _kd_state: Optional[_ModuleState] = dataclasses.field(
      repr=False,
      compare=False,
      hash=False,
      default=None,
  )

  if typing.TYPE_CHECKING:
    # Set by `traverse.recursive_set_parent`
    _kd_name: str = dataclasses.field(init=False)
    _kd_future_parent: Optional[traverse.Future[Module]] = dataclasses.field(
        init=False
    )

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    intermediate.setup_cls(cls)
    jax.tree_util.register_pytree_node_class(cls)

  def __post_init__(self, **kwargs):
    super().__post_init__(**kwargs)
    object.__setattr__(self, '_kd_init_finished', True)

  def init_bind(
      self: _SelfT,
      rng: int | jax.Array | dict[str, jax.Array],
      *args,
      streams: tuple[str, ...] = (Collection.DROPOUT,),
      **kwargs,
  ) -> _SelfT:
    """Initialize the module, returning a binded version."""
    # Note: Only top-level has a state. Childs recover the state from the
    # parent/scope.
    self._kd_state = _ModuleState(
        params=None,
        streams=streams,
        rngs={},
    )

    # Normalize the args/kwargs
    # Is it possible to have non-array kwargs ?
    args, kwargs = etree.spec_like((args, kwargs), ignore_other=False)
    args, kwargs = etree.map(_as_empty, (args, kwargs))

    # Set the childs `_kd_parent`
    self = self._replace_state()  # pylint: disable=self-cls-assignment

    # Generate the rngs for initialization
    rngs = _normalize_rngs(rng, streams=streams, add_params=True)

    # Initialize the state
    with context.set_in_call_state(self):
      variables = flax.core.freeze(
          self.init(
              rngs,
              *args,
              **kwargs,
              mutable=(Collection.PARAMS, Collection.INTERMEDIATES),
          )
      )

    return self._replace_state(params=variables.get(Collection.PARAMS, {}))

  @_bind_only_method
  def with_rng(
      self: _SelfT,
      rng: int | jax.Array | dict[str, jax.Array] | None = None,
  ) -> _SelfT:
    """Replace the rngs keys.

    Can be called:

    * `model = model.with_rng()`: Replace key with next key
    * `model = model.with_rng(0)`: Create a key from the seed.
    * `model = model.with_rng(key)`: Key distributed among streams
    * `model = model.with_rng({'dropout': key})`: streams explicitly defined

    Args:
      rng: Random key.

    Returns:
      The updated model with the next key.
    """
    # When rng is None, auto-increment rng
    if rng is None:
      # TODO(epot): When LazyRng, should instead increment rng counter to match
      # flax behavior, but difficult because we do not have access to flax
      # internal counter
      rng = {name: k.next() for name, k in self.rngs.items()}
    if isinstance(rng, int):
      rng = random.PRNGKey(rng)
    return self._replace_state(
        rngs=_normalize_rngs(rng, streams=self._root_state.streams)
    )

  @property
  @_bind_only_method
  def rngs(self) -> dict[str, random.PRNGKey]:
    """Returns `dict[str, PRNGKey]` mapping key to."""
    if self._kd_state is None:
      rngs = self._root_state.rngs
      # Fold-in the random info
      rngs = {
          k: flax.core.scope.LazyRng.create(rng, *self._kd_parent_names)
          for k, rng in rngs.items()
      }
      return rngs
    else:
      return self._kd_state.rngs  # pytype: disable=bad-return-type

  @_bind_only_method
  def train(self: _SelfT) -> _SelfT:
    """Switch mode to training."""
    return self._replace_state(training=True)

  @_bind_only_method
  def eval(self: _SelfT) -> _SelfT:
    """Switch mode to evaluation (disable dropout,...)."""
    return self._replace_state(training=False)

  @property
  @_bind_only_method
  def training(self) -> bool:
    """Returns `True` if mode is training."""
    return self._root_state.training

  @property
  @_bind_only_method
  def params(self) -> flax.core.scope.FrozenVariableDict:
    """Model weights."""
    params = self._root_state.params
    for name in self._kd_parent_names:
      params = params.get(name, {})  # pytype: disable=attribute-error
    return params  # pytype: disable=bad-return-type

  @_bind_only_method
  def param_tree_on(self: _SelfT) -> _SelfT:
    """Makes `tree_utils` only act on params."""
    return self._replace_state(tree_params_only=True)

  @_bind_only_method
  def param_tree_off(self: _SelfT) -> _SelfT:
    """Makes `tree_utils` act on everything."""
    return self._replace_state(tree_params_only=False)

  def call_with_intermediates(
      self: _SelfT, *args: Any, **kwargs: Any
  ) -> tuple[Any, _SelfT]:
    """Call the module with intermediates.

    Wrapper around `__call__` which also return the intermediate values:

    ```
    y = model(x)

    y, intermediates = model.call_with_intermediates(x)
    ```

    The intermediate values have the same structure as the model.

    Args:
      *args: Arguments forwarded to `module.__call__`
      **kwargs: Arguments forwarded to `module.__call__`

    Returns:
      `module.__call__` output
      Intermediate values.
    """
    with self.capture_intermediates() as intermediates:
      return self(*args, **kwargs), intermediates

  # ========== Internal methods ==========

  @property
  @_skip_wrap_call
  def _kd_parent(self) -> Optional[Module]:
    """Returns the parent."""
    if self._kd_future_parent is not None:
      return self._kd_future_parent.value
    return None

  @property
  @_skip_wrap_call
  def _kd_parent_names(self) -> list[str]:
    """List of path (excluding the first)."""

    parent_names = []
    parent = self
    while parent._kd_parent is not None:  # pylint: disable=protected-access  # pytype: disable=attribute-error
      parent_names.append(parent._kd_name)  # pylint: disable=protected-access
      parent = parent._kd_parent  # pylint: disable=protected-access

    return list(reversed(parent_names))

  # Return type should be Optional[_ModuleState] but we would then loose
  # auto-complete.
  @property
  @_skip_wrap_call
  @_bind_only_method
  def _root_state(self) -> _ModuleState:
    """Returns the root parent state."""
    # Inside a call context, we directly get the state
    # Indeed, modules defined inside `nn.compact` are not availabe
    if context.in_call_state:
      return context.in_call_state
    parent = self
    while parent._kd_parent is not None:  # pylint: disable=protected-access  # pytype: disable=attribute-error
      parent = parent._kd_parent  # pylint: disable=protected-access
    return parent._kd_state  # pylint: disable=protected-access  # pytype: disable=bad-return-type

  @property
  @_skip_wrap_call
  def _is_bind(self) -> bool:
    return hasattr(self, '_kd_name') or bool(context.in_call_state)

  @_skip_wrap_call
  def _replace_state(self: _SelfT, **kwargs) -> _SelfT:
    """Recursivelly update all the childs parents."""
    # TODO(epot): Support attributes defined in `.setup()`
    if self._kd_state is None:
      new_state_kwargs = dict(
          params=self.params,
          rngs=self.rngs,
      )
      new_state_kwargs.update(kwargs)
      new_state = self._root_state.replace(**new_state_kwargs)
    else:
      new_state = self._kd_state.replace(**kwargs)  # pytype: disable=attribute-error

    # First update the state
    new_self = dataclasses.replace(self, _kd_state=new_state)

    # Recursivelly update the modules to link to the new parents
    new_self = traverse.recursive_set_parent(new_self)

    return new_self

  @contextlib.contextmanager
  def capture_intermediates(self: _SelfT) -> Iterator[_SelfT]:
    """Track the intermediate values.

    Note that this function isn't meant to be called directly but instead
    through `y, intermediates = model.call_and_capture(x)`.

    Usage:

    ```python
    with model.capture_intermediates() as intermediates:
      y = model(x)  # Model set `model.xxx`

    # After the contextmanager end, `intermediates` contain the captured
    # intermediate values.
    intermediates.xxx
    ```

    Yields:
      The module proxy containing the intermediate values

    Raises:
      RuntimeError: If contextmanager are nested.
    """

    proxy = intermediate_proxy.ModuleIntermediateProxy(self)
    if context.capture_proxy:
      raise RuntimeError('`capture_intermediates()` calls cannot be nested.')
    try:
      context.capture_proxy = proxy
      yield proxy  # pytype: disable=bad-return-type
    finally:
      proxy._finalize()  # pylint: disable=protected-access
      context.capture_proxy = None

  @_bind_only_method
  def tree_flatten(
      self,
  ) -> tuple[list[flax.core.scope.FrozenVariableDict], Module]:
    """`jax.tree_utils` support."""
    if not self._kd_state:
      self = self._replace_state()  # Detach the child module  # pylint: disable=self-cls-assignment
    if self._kd_state.tree_params_only:
      vals = [self._kd_state.params]
    else:
      vals = [self._kd_state.params, self._kd_state.rngs]
    return (vals, self)  # pytype: disable=bad-return-type

  @classmethod
  def tree_unflatten(
      cls: type[_SelfT],
      metadata: Module,
      array_field_values: list[flax.core.scope.FrozenVariableDict],
  ) -> _SelfT:
    assert metadata._kd_state  # pylint: disable=protected-access
    if metadata._kd_state.tree_params_only:  # pylint: disable=protected-access
      (params,) = array_field_values
      return metadata._replace_state(params=params)  # pylint: disable=protected-access
    else:
      (params, rngs) = array_field_values
      return metadata._replace_state(params=params, rngs=rngs)  # pylint: disable=protected-access

  @_skip_wrap_call
  def _call_wrapped_method(self, fn, args, kwargs):
    """All function calls (`__call__`,...)."""

    # No-op for `Module` functions
    # TODO(epot): Better heuristic ?
    if fn.__module__ == 'kauldron.klinen.module':
      return super()._call_wrapped_method(fn, args, kwargs)

    # In-call set: Use default flax behavior
    if context.in_call_state:
      return super()._call_wrapped_method(fn, args, kwargs)

    if not self._is_bind:
      try:
        return super()._call_wrapped_method(fn, args, kwargs)
      except flax.errors.CallCompactUnboundModuleError:  # pylint: disable=try-except-raise
        raise
      else:
        raise RuntimeError("Calling without scope didn't raise unbound error")

    state = self._kd_state  # pylint: disable=protected-access

    if not state:
      # No scope: binding not called (module non-initialized)
      if self._root_state is None:
        # Call original method, to raise flax `CallCompactUnboundModuleError`
        try:
          return super()._call_wrapped_method(fn, args, kwargs)
        except flax.errors.CallCompactUnboundModuleError:  # pylint: disable=try-except-raise
          raise
        else:
          raise RuntimeError("Calling without scope didn't raise unbound error")
      else:
        # Detach module: e.g. `model.encoder(x)`
        return getattr(self._replace_state(), fn.__name__)(*args, **kwargs)
    elif state.params is None:
      # Should never happens in the `model.init()` function
      raise RuntimeError('Module not initialized.')
    else:
      # Top-level bind call
      with context.set_in_call_state(self):
        y, variables = self.apply(
            {Collection.PARAMS: state.params},
            rngs=state.rngs,
            method=getattr(self, fn.__name__),
            *args,
            **kwargs,
            mutable=(Collection.INTERMEDIATES,),
        )
        if context.capture_proxy:
          context.capture_proxy._bind(  # pylint: disable=protected-access
              module=self,
              intermediate_dict=variables.get(Collection.INTERMEDIATES, {}),
          )
        return y

    raise RuntimeError('Should have returned before.')

  if not typing.TYPE_CHECKING:

    @_skip_wrap_call
    def __getattr__(self, name: str) -> Any:
      maybe_descriptor = getattr(type(self), name, None)
      if isinstance(maybe_descriptor, intermediate.IntermediateDescriptor):
        # If `__get__` raise `AttributeError`, getattr will be called so
        # explicitly call `__get__` a second time to propagate the error.
        maybe_descriptor.__get__(self, type(self))
      else:  # Default flax behavior
        super().__getattr__(name)

    @_skip_wrap_call
    def __setattr__(self, name: str, value: Any) -> None:
      maybe_descriptor = getattr(type(self), name, None)
      if isinstance(maybe_descriptor, intermediate.IntermediateDescriptor):
        # Bypass flax setattr to use the descriptor
        object.__setattr__(self, name, value)
      else:  # Default flax behavior
        super().__setattr__(name, value)


def _as_empty(arr: enp.ArraySpec) -> jax.Array:
  """Create empty array."""
  # Downcast float64 to float32 to avoid Jax warning. Or better that the
  # user is aware of it ?
  dtype = np.float32 if arr.dtype == np.float64 else arr.dtype
  return jnp.empty(shape=arr.shape, dtype=dtype)


def _normalize_rngs(
    rng: int | jax.Array | dict[str, jax.Array],
    streams: list[str] | tuple[str, ...],
    add_params: bool = False,
) -> dict[str, jax.Array]:
  """Normalize the rngs keys."""
  rng = jax.tree.map(random.PRNGKey, rng)
  if isinstance(rng, dict):
    return rng
  elif isinstance(rng, random.PRNGKey):
    # Could we collect the streams from the childs modules ? Difficult as
    # some modules are only available inside `__call__`.
    rngs = {name: rng.fold_in(name) for name in streams}
    if add_params:
      # Do not `fold_in('params')` so `.init_bind(key)` is consistent with
      # `.init(key)`
      rngs[Collection.PARAMS] = rng
    return rngs
  else:
    raise TypeError(f'Unexpected key {rng}')
