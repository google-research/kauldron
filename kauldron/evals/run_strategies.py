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

"""Run strategies.

The `RunStrategy` are defined in the Kauldron konfig, but also imported in the
KXM launcher (to detect which eval jobs to launch).
"""

from __future__ import annotations

import dataclasses

from kauldron.xm._src import job_params


class RunStrategy:
  """Base class for info on how to run the evaluation.

  `RunStrategy` are divided into:

  * Strategies which run along train (in the same XM job):
    * `EveryNSteps`: Run evaluation every `X` steps
    * `Once`: Run a single evaluation after `X` steps
  * Strategies which run in a separate XManager job:
    * `StandaloneEveryCheckpoint`: Run evaluation every time a new checkpoint is
      found. Note that if eval is too slow, itermediate checkpoints will be
      skipped.
    * `StandaloneLastCheckpoint`: Only run evaluation once, after train has
      completed.

  Evaluators run in a standalone job can be grouped together through the
  `job_group='group_name'` attribute. This allow to save resources by sharing
  the same job for multiple evaluators.

  Example:

  ```python
  shared_run = kd.evals.StandaloneEveryCheckpoint(
      job_group='separate',
      # Standalone evaluators supports all `kxm.Job` parameters.
      platform='a100=1',
  ))

  cfg.evals = {
      'eval0': kd.evals.Evaluator(run=kd.evals.EveryNSteps(100)),
      'eval1': kd.evals.Evaluator(run=shared_run),
      'eval2': kd.evals.Evaluator(run=shared_run),
  }
  ```

  The XManager experiment will containing 2 tasks: `train` (running `eval0`)
  and `separate` (running `eval1` and `eval2`).

  Those objects are never resolved by the Kauldron Trainer config (to avoid
  XManager dependency).

  Currently, the Kauldron XM launcher hardcode assumptions on the classes here.
  Contact us if you need more flexibility and for custom run behavior.
  """

  def should_eval_in_train(self, step: int) -> bool:
    """Whether the evaluator should be run for the given train-step."""
    raise NotImplementedError()


@dataclasses.dataclass(kw_only=True, frozen=True)
class AlongTrain(RunStrategy):
  """Run eval inside the same train XM job."""


@dataclasses.dataclass(kw_only=True, frozen=True)
class Standalone(RunStrategy, job_params.JobParams):
  """Run eval in a separate standalone XM job.

  Attributes:
    job_group: The name of the job group. If set, all the evaluators with the
      same `job_group` will share the same XManager job.  Otherwise, each
      evaluator will run a separate job.
  """

  job_group: str | None = None

  # Do not resolve any of the `JobParams` arguments (so we do not need to import
  # XManager from the Kauldron side)
  __konfig_resolve_exclude_fields__ = tuple(
      f.name for f in dataclasses.fields(job_params.JobParams)
  )

  def should_eval_in_train(self, step: int) -> bool:
    return False  # Never run `Standalone` inside the `train` job


@dataclasses.dataclass(frozen=True)
class EveryNSteps(AlongTrain):
  """Run eval every N train steps."""

  n: int

  def should_eval_in_train(self, step: int) -> bool:
    return step % self.n == 0


@dataclasses.dataclass(frozen=True)
class Once(AlongTrain):
  """Run eval only after the `XX` train steps."""

  step: int

  def should_eval_in_train(self, step: int) -> bool:
    """Whether the evaluator should be run for the given train-step."""
    return step == self.step


@dataclasses.dataclass(kw_only=True, frozen=True)
class StandaloneEveryCheckpoint(Standalone):
  """Run eval continuously everytime a new checkpoint is found.

  If the eval takes too long, intermediate checkpoints might be skipped.

  Run as a separate XM job. All `kxm.Job` parameters are optionally supported.

  Example:

  ```python
  kd.evals.Evaluator(
      run=kd.evals.StandaloneEveryCheckpoint(platform='a100=1'),
  )
  ```

  If `job_group='group_name'`, all the evaluators sharing the same `job_group`
  will share the same XManager job (to save resources).
  """


@dataclasses.dataclass(kw_only=True, frozen=True)
class StandaloneLastCheckpoint(Standalone):
  """Run eval only after the last checkpoint, after train has completed.

  Run as a separate XM job. All `kxm.Job` parameters are optionally supported.

  Example:

  ```python
  kd.evals.Evaluator(
      run=kd.evals.StandaloneLastCheckpoint(platform='a100=1'),
  )
  ```

  If `job_group='group_name'`, all the evaluators sharing the same `job_group`
  will share the same XManager job (to save resources).
  """
