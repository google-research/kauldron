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

import jax
from kauldron.cli import cmd_utils
from kauldron.data import utils as data_utils
from xmanager import xm


def _parse_platform(value: str) -> tuple[str, str]:
  """Parse a `<platform>=<topology>` string (e.g. `'pf=2x2'`).

  Uses `xm.ResourceType` for alias resolution (e.g. 'pf' -> 'PUFFERFISH').
  """
  parts = value.split('=', maxsplit=1)
  if len(parts) != 2 or not parts[0] or not parts[1]:
    raise ValueError(
        f"Expected '<platform>=<topology>' (e.g. 'pf=2x2'), got {value!r}"
    )
  name, topology = parts
  platform = xm.ResourceType[name].name.title()
  return platform, topology


@dataclasses.dataclass(frozen=True, kw_only=True)
class EvalShape(cmd_utils.SubCommand):
  """Run a train step through jax.eval_shape (shapes only, no compute)."""

  def __call__(self) -> str:
    trainer = self.trainer
    elem_spec = trainer.train_ds.element_spec
    elem_sharding = trainer.sharding.batch

    init_fn = functools.partial(trainer.init_state, skip_transforms=True)
    state_spec = jax.eval_shape(init_fn)

    m_batch = data_utils.mock_batch_from_elem_spec(elem_spec, elem_sharding)
    context_spec = jax.eval_shape(
        trainer.trainstep.step, batch=m_batch, state=state_spec
    )

    outputs = ['eval_shape: OK']
    outputs.append(f'state_spec: {state_spec}')
    outputs.append(f'context_spec: {context_spec}')
    return '\n'.join(outputs)


@dataclasses.dataclass(frozen=True, kw_only=True)
class MockTpu(cmd_utils.SubCommand):
  """Run a train step on a simulated TPU (shapes and sharding, no real compute)."""

  platform: str = 'pf=2x2'

  def __call__(self) -> str:
    import importlib  # pylint: disable=g-import-not-at-top

    platform_name, topology = _parse_platform(self.platform)
    mb = importlib.import_module(
        'learning.deepmind.jax.mock_backend.mock_backend'
    )
    mb.use_mock_backend(  # pytype: disable=attribute-error
        platform=platform_name,
        topology=topology,
    )
    trainer = self.trainer
    trainer.train()
    return 'mock_tpu: OK'


@dataclasses.dataclass(frozen=True, kw_only=True)
class Cpu(cmd_utils.SubCommand):
  """Run a train step on CPU (real compute, slowest)."""

  def __call__(self) -> str:
    jax.config.update('jax_platforms', 'cpu')
    trainer = self.trainer
    trainer.train()
    return 'cpu: OK'


_SUBCOMMANDS = {
    'eval_shape': EvalShape,
    'mock_tpu': MockTpu,
    'cpu': Cpu,
}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Run(cmd_utils.CommandGroup):
  """Run commands for local training validation."""

  sub_command: Union[EvalShape, MockTpu, Cpu] = dataclasses.field(
      metadata={'subparsers': _SUBCOMMANDS}
  )
