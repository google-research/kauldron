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
from kauldron.inspect import inspect as inspect_lib
import tensorflow_datasets as tfds

# MARK: EvalShape


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


class _MockIterator:
  """Iterator that yields the same mock batch, optionally limited."""

  def __init__(self, batch, spec, max_items=None):
    self.batch = batch
    self.element_spec = spec
    self._max_items = max_items
    self._count = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self._max_items is not None and self._count >= self._max_items:
      raise StopIteration
    self._count += 1
    return self.batch


class _MockDataset:
  """Dataset that yields mock batches without reading real data."""

  def __init__(self, batch, spec, batch_size, max_items=None):
    self.batch = batch
    self.element_spec = spec
    self.batch_size = batch_size
    self._max_items = max_items

  def __iter__(self):
    return _MockIterator(self.batch, self.element_spec, self._max_items)

  def take(self, n):
    """Return a copy limited to n items."""
    return _MockDataset(self.batch, self.element_spec, self.batch_size, n)

  def cache(self):
    """Return self; mock data is already in memory."""
    return self

  def device_put(self, sharding):
    """Return self; mock data is already on device."""
    del sharding
    return self


def _replace_ds_with_mock(obj, attr_name: str) -> None:
  """Replace a dataset attribute on obj with a _MockDataset."""
  ds = getattr(obj, attr_name)
  elem_spec = ds.element_spec
  batch_size = getattr(ds, 'batch_size', 1)

  def _spec_to_zeros(spec):
    shape = tuple(d if d is not None else 1 for d in spec.shape)
    return jax.numpy.zeros(shape, spec.dtype)

  m_batch = jax.tree.map(_spec_to_zeros, elem_spec)
  mock_ds = _MockDataset(m_batch, elem_spec, batch_size)
  object.__setattr__(obj, attr_name, mock_ds)


def _patch_readout_trainer_for_step(evaluator) -> None:
  """Monkey-patch readout_trainer_for_step to mock datasets on the result."""
  original_fn = evaluator.readout_trainer_for_step

  def patched_fn(state, step):
    inner_trainer = original_fn(state, step)
    with jax.transfer_guard('allow'):
      _replace_ds_with_mock(inner_trainer, 'train_ds')
      for inner_eval in inner_trainer.evals.values():
        if hasattr(inner_eval, 'ds'):
          _replace_ds_with_mock(inner_eval, 'ds')
    return inner_trainer

  object.__setattr__(evaluator, 'readout_trainer_for_step', patched_fn)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Train(cu.SubCommand):
  """Run trainer.train()."""

  mock_data: bool = True

  def __call__(self) -> None:
    self.print_config_origin()

    cfg = self.cfg
    assert cfg is not None
    # Ensure exactly one training step is performed to match kd_test.ipynb
    cfg.stop_after_steps = 1  # pyrefly: ignore[missing-attribute]

    with kd.konfig.imports():
      from kauldron.train import metric_writer  # pylint: disable=g-import-not-at-top

      cfg.writer = metric_writer.NoopWriter()  # pyrefly: ignore[missing-attribute]

    if hasattr(cfg, 'evals'):
      kd.kontext.set_by_path(cfg, 'evals.**.num_batches', 1)
      kd.kontext.set_by_path(cfg, 'evals.**.stop_after_steps', 1)
      for _, eval_cfg in cfg.evals.items():
        if 'readout_config' in eval_cfg:
          eval_cfg.stop_after_steps = 1

    if self.mock_data:
      # Resolve config under tfds.testing.mock_data() so that
      # element_spec is available without reading real data from CNS.
      with tfds.testing.mock_data(num_examples=1):
        trainer = self.trainer  # trigger config resolution
      print('Using mock data!')
      _replace_ds_with_mock(trainer, 'train_ds')
      # Monkey-patch TrainEvaluator.readout_trainer_for_step to also
      # mock the inner trainer's datasets after construction.
      for evaluator in trainer.evals.values():
        if hasattr(evaluator, 'readout_trainer_for_step'):
          _patch_readout_trainer_for_step(evaluator)
    else:
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

  mock_data: bool = True

  def __call__(self) -> None:
    self.print_config_origin()

    cfg = self.cfg
    assert cfg is not None
    with kd.konfig.imports():
      from kauldron.train import metric_writer  # pylint: disable=g-import-not-at-top

      cfg.writer = metric_writer.NoopWriter()  # pyrefly: ignore[missing-attribute]

    if hasattr(cfg, 'evals'):
      kd.kontext.set_by_path(cfg, 'evals.**.num_batches', 1)
      kd.kontext.set_by_path(cfg, 'evals.**.stop_after_steps', 1)
      for _, eval_cfg in cfg.evals.items():
        if 'readout_config' in eval_cfg:
          eval_cfg.stop_after_steps = 1

    if self.mock_data:
      with tfds.testing.mock_data(num_examples=1):
        trainer = self.trainer  # trigger config resolution
      print('Using mock data!')
      # We don't have train_ds in eval command; we mostly care about evaluator
      # datasets.
      for evaluator in trainer.evals.values():
        if hasattr(evaluator, 'readout_trainer_for_step'):
          _patch_readout_trainer_for_step(evaluator)
    else:
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


# MARK: EstimateHBM

# MARK: Run

_SUBCOMMANDS = {
    'eval_shape': EvalShape,
    'train': Train,
    'eval': Eval,
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Run(cu.CommandGroup):
  """Run commands for local training validation."""

  sub_command: Union[EvalShape] = dataclasses.field(
      metadata={'subparsers': _SUBCOMMANDS}
  )
