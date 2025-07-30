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

"""Various utils."""

from collections.abc import Iterable, Iterator
import functools
import itertools
from typing import Optional, TypeVar
from jax.experimental import checkify
from tqdm import auto as tqdm

_T = TypeVar('_T')


def json_list_to_tuple(json_value):
  """Normalize the `json` to use `tuple` rather than `list`."""
  match json_value:
    case dict():
      return {k: json_list_to_tuple(v) for k, v in json_value.items()}
    case list():
      return tuple(json_list_to_tuple(v) for v in json_value)
    case _:
      return json_value


def enum_iter(
    iter: Iterable[_T],  # pylint: disable=redefined-builtin
    *,
    init_step: int = 0,
    total_steps: Optional[int] = None,
    desc: Optional[str] = None,
    log_xm: bool = False,
    **tqdm_kwargs,
) -> Iterator[tuple[int, _T]]:
  """Iterable wrapper.

  Args:
    iter: The iterable
    init_step: If the iterable is restored (e.g. after preemption). This does
      NOT skip elements in the iterable, just assume the first element yield has
      index `init_step`.
    total_steps: Last step (exclusive)
    desc: tqdm desciption
    log_xm: Whether to log to XManager
    **tqdm_kwargs: Arguments forwarded to TQDM.

  Yields:
    step_id
    elem
  """
  if total_steps is None or total_steps < 0:  # Infinite iterator
    range_ = itertools.count(init_step)
    try:
      total = len(iter)  # pytype: disable=wrong-arg-types
    except TypeError:
      total = None
  else:
    range_ = range(init_step, total_steps)
    total = total_steps

  tqdm_class = tqdm.tqdm

  for i, ex in tqdm_class(
      zip(range_, iter),
      initial=init_step,
      total=total,
      desc=desc,
      **tqdm_kwargs,
  ):
    yield i, ex


def checkify_wrapper(fn):
  """Decorator to ignore checkify errors by default, but raise them if needed.

  The wrapped function will be checkified, but the errors will be ignored by
  default. This is useful to avoid crashing code that contains user-checks.

  The decorator adds an additional `raise_checkify_errors` argument to the
  function, which can be used to specify checkify error categories to raise.

  Args:
    fn: The function wrap (and checkify).

  Returns:
    The wrapped function.
  """

  @functools.wraps(fn)
  def _checkify_wrapper(
      *args,
      raise_checkify_errors: frozenset[checkify.ErrorCategory] = frozenset(),
      **kwargs,
  ):
    checked_fn = checkify.checkify(fn, errors=raise_checkify_errors)
    error, result = checked_fn(*args, **kwargs)
    if raise_checkify_errors:  # else ignore the errors
      checkify.check_error(error)
    return result

  return _checkify_wrapper
