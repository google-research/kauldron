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

The `RunStrategy` implementation is more distributed due to technical
constraints: There's 2 configurations (kxm launcher and Kauldron trainer)
which cannot be resolved together (XManager cannot import flax,... while
Kauldron cannot import the full XM API).

Instead, the `RunStrategy` object only live on the XManager side (can be
resolved safely in kxm). On Kauldron side, kauldron can use
`run_strategies.run_strategy_cfg_to_kauldron_run_info()` to safely resolve the
strategy in an object that can be used on the Kauldron side.
"""

from __future__ import annotations

import dataclasses

from kauldron import konfig
from kauldron.xm._src import job_params


class RunStrategy:
  """Base class for info on how to run the evaluation.

  Only the following `RunStrategy` are implemented:

  * `RunEvery`: Run evaluation along train, every `X` steps
  * `RunOnce`: Run a single evaluation along train after `X` steps
  * `RunXM`: Run a single evaluation in a separate `XManager` job.
  * `RunSharedXM`: Run multiple evaluations in a separate `XManager` job.

  Currently, the Kauldron XM launcher hardcode assumptions on the classes here.
  Contact us if you need more flexibility and for custom run behavior.
  """


@dataclasses.dataclass(frozen=True)
class RunEvery(RunStrategy):
  """Run eval every `XX` train steps."""

  every: int


@dataclasses.dataclass(frozen=True)
class RunOnce(RunStrategy):
  """Run eval only after the `XX` train steps."""

  step: int


@dataclasses.dataclass(frozen=True)
class RunXM(RunStrategy, job_params.JobParams):
  """Run eval on a separate XManager job.

  Run continuously everytime a new is checkpoint is found.

  Example:

  ```python
  cfg.evals = {
      'eval0': kd.evals.Evaluator(run=kd.evals.RunXM()),
      'eval1': kd.evals.Evaluator(run=kd.evals.RunXM()),
  }
  ```

  The experiment will containing 3 tasks: `train`, `eval0`, `eval1`.
  """


@dataclasses.dataclass(frozen=True)
class RunSharedXM(RunXM):
  """Run multiple evals on a unique XManager job.

  It's possible to have a single task for multiple evaluators.

  ```python
  run_xm = kd.evals.RunSharedXM(shared_name='shared_eval')
  cfg.evals = {
      # The 2 evals share the same XManager task
      'eval_train': kd.evals.Evaluator(run=run_xm),
      'eval_eval': kd.evals.Evaluator(run=run_xm),
  }
  ```

  In which case the experiment will contain 2 jobs: `train` and `shared_eval`

  Attributes:
    shared_name: str
    final_eval: bool
  """

  shared_name: str
  final_eval: bool = False


@dataclasses.dataclass(frozen=True)
class KauldronRunStrategy:
  """Kauldron implementation of the run-strategies."""

  def should_eval_in_train(self, step: int) -> bool:
    """Whether the evaluator should be run for the given train-step."""
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class KauldronRunEvery(KauldronRunStrategy):
  """Kauldron implementation of the `RunEvery`."""

  every: int

  def should_eval_in_train(self, step: int) -> bool:
    """Whether the evaluator should be run for the given train-step."""
    return step % self.every == 0


@dataclasses.dataclass(frozen=True)
class KauldronRunOnce(KauldronRunStrategy):
  """Kauldron implementation of the `RunOnce`."""

  step: int

  def should_eval_in_train(self, step: int) -> bool:
    """Whether the evaluator should be run for the given train-step."""
    return step == self.step


@dataclasses.dataclass(frozen=True)
class KauldronRunNoOp(KauldronRunStrategy):
  """Kauldron implementation of the `RunXM`, `RunSharedXM`."""

  def should_eval_in_train(self, step: int) -> bool:
    """Whether the evaluator should be run for the given train-step."""
    return False


def run_strategy_cfg_to_kauldron_run_info(
    run: konfig.ConfigDictLike[RunStrategy],
) -> KauldronRunStrategy:
  """Resolve the kxm object into Kauldron one."""
  # Cannot resolve `RunStrategy` to avoid importing the full XManager API
  if not isinstance(run, konfig.ConfigDict):
    raise TypeError(
        f'Unexpected run type. Expected `konfig.ConfigDict`. Got: {run}'
    )
  # This is where the convertion `kxm` <> `Kauldron` happen while avoiding
  # fully resolving the `XManager`
  if run.__qualname__.endswith(('.RunEvery', ':RunEvery')):
    # Rewrite the import to avoid a full import of the Kauldron codebase
    run.__qualname__ = 'kauldron.xm._src.run_strategies:RunEvery'
    run = konfig.resolve(run)
    return KauldronRunEvery(every=run.every)
  elif run.__qualname__.endswith(('.RunOnce', ':RunOnce')):
    # Rewrite the import to avoid a full import of the Kauldron codebase
    run.__qualname__ = 'kauldron.xm._src.run_strategies:RunOnce'
    run = konfig.resolve(run)
    return KauldronRunOnce(step=run.step)
  elif run.__qualname__.endswith('.RunXM'):
    return KauldronRunNoOp()
  elif run.__qualname__.endswith('.RunSharedXM'):
    return KauldronRunNoOp()
  else:
    raise TypeError(
        f'Unexpected eval run strategy. Got: {run} ({run.__qualname__!r})'
    )
