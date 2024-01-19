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

"""Jit utils."""

from collections.abc import Callable, Iterable
import functools
import inspect
from typing import Any, TypeVar

import jax
from jax._src import sharding_impls

_Fn = TypeVar('_Fn')


def jit(
    *,
    in_shardings=sharding_impls.UNSPECIFIED,
    out_shardings=sharding_impls.UNSPECIFIED,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnames: str | Iterable[str] | None = None,
    **kwargs: Any,
) -> Callable[[_Fn], _Fn]:
  """Wrapper around `jax.jit`.

  Differences:

  * Lazily compute `in_sharding` / `out_sharding` (otherwise fail due to
    sharding impossible to create before `app.run(main)` call)
  * `in_sharding` only check the sharding was set outside.
  * Supports `kwargs` for `in_sharding` (
    https://github.com/google/jax/issues/17400)
  * Is called without `@functools.partial`

  See `jax.jit` doc:
  https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html

  Args:
    in_shardings:
    out_shardings:
    static_argnames:
    donate_argnames:
    **kwargs: Additional kwargs passed to `@jax.jit`

  Returns:
    decorator: The `@jax.jit` decorator
  """

  if in_shardings != sharding_impls.UNSPECIFIED:
    raise NotImplementedError(
        'Due to performance issues, `in_shardings=` is disabled now. Please '
        'use `kd.sharding.device_put()` before calling your model.'
    )

  def _jit_fn(fn):
    nonlocal in_shardings, out_shardings
    if callable(in_shardings):
      in_shardings = in_shardings()
    if callable(out_shardings):
      out_shardings = out_shardings()
    jitted_fn = functools.partial(
        jax.jit,
        # TODO(epot): Restore kwargs:
        # https://github.com/google/jax/issues/17400
        # in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnames=static_argnames,
        donate_argnames=donate_argnames,
        **kwargs,
    )(
        fn  # pylint: disable=too-many-function-args
    )

    # TODO(epot): Remove hack once https://github.com/google/jax/issues/17400
    # is supported
    # `in_sharding` do not support kwargs, so manually shard
    if in_shardings is sharding_impls.UNSPECIFIED:
      return jitted_fn

    sig = inspect.signature(fn)

    @functools.wraps(jitted_fn)
    def wrapped_fn(*args, **kwargs):
      args = sig.bind(*args, **kwargs)
      # TODO(epot): Could be more flexible. Currently it fragile for
      # more complex sharding (single value, or different values for tree...)
      for k, v in in_shardings.items():
        assert k in args.arguments, f'Missing arg: {k}'

        # pylint: disable=cell-var-from-loop
        def _assert_sharding(path, value):
          if not hasattr(value, 'sharding'):
            raise ValueError(
                f'Expected argument {k!r} to be jax array: {type(value)} for'
                f' {path=}'
            )
          if value.sharding != v:
            raise ValueError(
                f'Unexpected sharding for arg {k!r}: '
                f'{value.sharding} != {v} for {path=}'
            )

        # pylint: enable=cell-var-from-loop

        jax.tree_util.tree_map_with_path(_assert_sharding, args.arguments[k])
      return jitted_fn(*args.args, **args.kwargs)

    return wrapped_fn

  def decorator(fn):
    fn.__kd_jitted__ = None

    @functools.wraps(fn)
    def lazy_jit_initialization(*args, **kwargs):
      # Post-pone the `@jax.jit` call when the function is first called
      # This makes sure `app.run` has been called.
      if fn.__kd_jitted__ is None:
        fn.__kd_jitted__ = _jit_fn(fn)

      return fn.__kd_jitted__(*args, **kwargs)

    return lazy_jit_initialization

  return decorator
