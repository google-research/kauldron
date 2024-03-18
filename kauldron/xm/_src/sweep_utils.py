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

"""Sweep lib."""

from __future__ import annotations

import abc
import dataclasses
import functools
from typing import Any, Callable, Iterable, Self

from kauldron.xm._src import jobs_info

_SweepKwargs = dict[str, Any]
_SweepKwargsGenerator = Callable[[], Iterable[_SweepKwargs]]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SweepInfo(abc.ABC):
  """Sweep information (how many sweep, sweep kwargs,...).

  Examples of implementations:

  * `kxm.SimpleSweep`: Explicitly provide the sweep kwargs to use
  * `kxm.KauldronSweep`: Load the sweep from the `def sweep()` from
    the `config.py`

  Sweep is applied internally by the orchestrator, like:

  ```python
  sweep_info = kxm.SimpleSweep([
      {'batch_size': 32},
      {'batch_size': 64},
  ])
  for sweep_item in sweep_info:
    xp.add(job, args=sweep_item.job_kwargs)
  ```
  """

  # Automatically set in `Experiment.__post_init__`
  # `_sweep_value` contains the value passed to `--xp.sweep=`
  _sweep_value: bool | str | list[str] = dataclasses.field(  # pytype: disable=annotation-type-mismatch
      default=None,
      repr=False,
  )

  @abc.abstractmethod
  def __iter__(self) -> Iterable[SweepItem]:
    """Iterate over the individual sweep."""
    raise NotImplementedError("Abstract method")

  def __len__(self) -> int:
    it = iter(self)  # pytype: disable=wrong-arg-types
    return len(list(it))  # Could be cached, but likely don't matter

  @functools.cached_property
  def tags(self) -> list[str]:
    """XM tags when sweep is activated."""
    return ["🧹"]

  def replace_with_jobs_provider(
      self, jobs_provider: jobs_info.JobsProvider
  ) -> Self:
    """Allow the sweep to access the job provider."""
    del jobs_provider
    return self


@dataclasses.dataclass(frozen=True, kw_only=True)
class NoSweep(SweepInfo):
  """No-op sweep."""

  def __iter__(self) -> Iterable[SweepItem]:
    yield SweepItem({})

  @functools.cached_property
  def tags(self) -> list[str]:
    return []


# pytype: disable=invalid-function-definition
@dataclasses.dataclass(frozen=True)
class SimpleSweep(SweepInfo):
  # pytype: enable=invalid-function-definition
  """Simple sweep (e.g. on Colab).

  Usage:

  ```python
  xp = kxm.Experiment(
      ...,
      sweep_info=kxm.SimpleSweep(...),
  )
  xp.launch()
  ```

  `SimpleSweep` accept either an explicit kwargs list, or a generator function:

  ```python
  kxm.SimpleSweep([
      {'batch_size': 32},
      {'batch_size': 64},
  ])
  ```

  ```python
  def my_sweep_fn():
    for batch_size in [32, 64]:
      yield {'batch_size': batch_size}

  sweep_info = kxm.SimpleSweep(my_sweep_fn)
  ```

  Attributes:
    items: Either a list of `dict[str, Any]` sweep kwargs, or a function that
      generate kwargs.
  """

  items: Iterable[_SweepKwargs] | _SweepKwargsGenerator
  _: dataclasses.KW_ONLY

  def __post_init__(self):
    items = self.items() if callable(self.items) else self.items
    object.__setattr__(self, "items", list(items))  # Allow to re-iterate

  def __iter__(self) -> Iterable[SweepItem]:
    for item in self.items:
      yield SweepItem(item)


@dataclasses.dataclass(frozen=True)
class SweepItem:
  """Single sweep (work-unit) argument info.

  Attributes:
    job_kwargs: Args to pass to the job
    xm_ui_kwargs: Args displayed in the XManager UI (see:
     )
  """

  job_kwargs: _SweepKwargs
  _: dataclasses.KW_ONLY
  xm_ui_kwargs: _SweepKwargs = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if not self.xm_ui_kwargs:
      object.__setattr__(self, "xm_ui_kwargs", self.job_kwargs.copy())
