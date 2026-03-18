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

"""Run-related CLI commands for local training validation."""

from __future__ import annotations

import dataclasses
import functools
from typing import Union

from etils import epy
import jax
import kauldron as kd
from kauldron.cli import cmd_utils as cu
from kauldron.data import utils as data_utils
import tensorflow_datasets as tfds


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvalShape(cu.SubCommand):
  """Run a train step through jax.eval_shape (shapes only, no compute)."""

  def __call__(self) -> None:
    self.print_config_origin()

    trainer = self.trainer  # trigger config resolution

    batch_size = getattr(trainer.train_ds, 'batch_size', 1)
    with (
        cu.timed('Getting element spec'),
        tfds.testing.mock_data(num_examples=batch_size),
    ):
      elem_spec = trainer.train_ds.element_spec
      elem_sharding = trainer.sharding.batch
      m_batch = data_utils.mock_batch_from_elem_spec(elem_spec, elem_sharding)

    with cu.timed('jax.eval_shape(trainer.init_state) '):
      # TODO(klausg): add skip_transforms as a CLI flag.
      init_fn = functools.partial(
          trainer.init_state,
          skip_transforms=True,
      )
      state_spec = jax.eval_shape(
          init_fn,
          # element_spec=elem_spec,
      )

    with cu.timed('jax.eval_shape(trainer.trainstep.step) '):
      train_state = jax.eval_shape(
          trainer.trainstep.step, batch=m_batch, state=state_spec
      )
      del train_state

    with cu.timed('trainer.context_specs'):
      context_spec = trainer.context_specs

    # TODO(klausg): this is very verbose. Add a CLI flag to control verbosity.
    print('context_specs: ', end='')
    cu.print_spec(context_spec)

    # TODO(klausg): could try to run the metrics computation too.


@dataclasses.dataclass(frozen=True, kw_only=True)
class Train(cu.SubCommand):
  """Run trainer.train()."""

  def __call__(self) -> None:
    self.print_config_origin()

    # Ensure exactly one training step is performed to match kd_test.ipynb
    self.cfg.stop_after_steps = 1

    if hasattr(self.cfg, 'evals'):
      kd.kontext.set_by_path(self.cfg, 'evals.**.num_batches', 1)

    trainer = self.trainer  # trigger config resolution

    with cu.timed('trainer.train()'):
      train_state, aux = trainer.train()
      del train_state, aux
    # We don't print the output of train() as it might be very verbose
    # but we can print that it succeeded.
    print('Successfully completed trainer.train()')

@dataclasses.dataclass(frozen=True, kw_only=True)
class Eval(cu.SubCommand):
  """Run trainer.eval()."""

  def __call__(self) -> None:
    self.print_config_origin()

    if hasattr(self.cfg, 'evals'):
      kd.kontext.set_by_path(self.cfg, 'evals.**.num_batches', 1)

    trainer = self.trainer  # trigger config resolution

    with cu.timed('trainer.init_state()'):
      state = trainer.init_state()

    eval_metrics = {}
    for name, evaluator in trainer.evals.items():
      with cu.timed(f'evaluator.evaluate({name})'):
        eval_metrics[name] = evaluator.evaluate(state=state, step=0)

    if eval_metrics:
      print('Evaluator metrics:')
      epy.pprint(eval_metrics)


_SUBCOMMANDS = {
    # Manually name the subcommand 'eval_shape' beacause simple-parsing
    # would turn this into 'evalshape' instead.
    'eval_shape': EvalShape,
    'train': Train,
    'eval': Eval,
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Run(cu.CommandGroup):
  """Run commands for local training validation."""

  sub_command: Union[EvalShape, Train, Eval] = dataclasses.field(
      metadata={'subparsers': _SUBCOMMANDS}
  )
