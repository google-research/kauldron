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

"""Various utils."""

from collections.abc import Iterable, Iterator
import itertools
from typing import Optional, TypeVar

import tqdm

_T = TypeVar('_T')


def enum_iter(
    iter: Iterable[_T],  # pylint: disable=redefined-builtin
    *,
    init_step: int = 0,
    total_steps: Optional[int] = None,
    desc: Optional[str] = None,
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

  for i, ex in tqdm.tqdm(
      zip(range_, iter),
      initial=init_step,
      total=total,
      desc=desc,
      **tqdm_kwargs,
  ):
    yield i, ex
