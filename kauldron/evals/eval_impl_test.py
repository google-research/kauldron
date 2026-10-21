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

from collections.abc import Iterator
import os
from typing import Any
from unittest import mock

from etils import epath
from kauldron import kd
from kauldron.evals import eval_impl
from examples import mnist_autoencoder
import pytest
import tensorflow_datasets as tfds


def test_eval_impl(tmp_path: epath.Path):
  # Load config and reduce size
  cfg = mnist_autoencoder.get_config()

  cfg.train_ds.batch_size = 1
  cfg.evals.eval.ds.batch_size = 1
  cfg.model.encoder.features = 3
  cfg.num_train_steps = 1
  cfg.workdir = os.fspath(tmp_path)

  with kd.konfig.mock_modules():
    for ev in cfg.evals.values():
      ev.run = kd.evals.StandaloneEveryCheckpoint()

  trainer = kd.konfig.resolve(cfg)

  def _mocked_iterator(**kwargs) -> Iterator[int]:
    del kwargs

    # Simulate train saving a step
    state = trainer.init_state()
    trainer.checkpointer.save(state, step=0)
    trainer.checkpointer.wait_until_finished()
    yield 0

  # Launch train
  with (
      tfds.testing.mock_data(),
      mock.patch(
          'orbax.checkpoint.checkpoint_utils.checkpoints_iterator',
          _mocked_iterator,
      ),
  ):

    # Write element_spec to disk so eval can use it in initialization
    trainer.writer.write_element_spec(0, trainer.train_ds.element_spec)

    aux = trainer.continuous_eval('eval')
    # Ensure at least one checkpoint was computed
    assert 'recon' in aux['eval'].loss_states


def _make_trainer(evaluate_side_effects: dict[str, Any]) -> mock.MagicMock:
  """Returns a fake trainer with a `StandaloneEveryCheckpoint` eval per key."""
  trainer = mock.MagicMock()
  trainer.setup.eval_only = False
  trainer.init_state.return_value = mock.Mock(step=0)
  trainer.evals = {}
  for name, side_effect in evaluate_side_effects.items():
    ev = mock.MagicMock()
    ev.name = name
    ev.run = mock.MagicMock(spec=kd.evals.StandaloneEveryCheckpoint)
    ev.evaluate.side_effect = side_effect
    trainer.evals[name] = ev
  return trainer


def _continuous_eval(
    trainer: mock.MagicMock, *, eval_names: list[str], steps: list[int]
) -> None:
  """Runs `continuous_eval` on fake checkpoints for the given steps."""
  states = [mock.Mock(step=step) for step in steps]
  with (
      mock.patch.object(eval_impl, '_get_element_spec'),
      mock.patch.object(
          eval_impl,
          '_preemptable_iter_new_checkpoints',
          return_value=iter(states),
      ),
      mock.patch.object(eval_impl, 'status'),
  ):
    eval_impl.continuous_eval(trainer, eval_names=eval_names)


def test_continuous_eval_does_not_rerun_failed_evaluator():
  trainer = _make_trainer({
      'healthy': None,
      'broken': RuntimeError('broken eval'),
  })

  with pytest.raises(ExceptionGroup, match='One or more') as exc_info:
    _continuous_eval(trainer, eval_names=['healthy', 'broken'], steps=[0, 1, 2])

  # The healthy eval runs on every checkpoint. The broken one only runs on the
  # first, so its error is reported once.
  healthy_steps = [
      call.kwargs['step']
      for call in trainer.evals['healthy'].evaluate.call_args_list
  ]
  assert healthy_steps == [0, 1, 2]
  trainer.evals['broken'].evaluate.assert_called_once()
  assert len(exc_info.value.exceptions) == 1
  assert exc_info.value.exceptions[0].__notes__ == [
      "Evaluator 'broken' failed at step 0"
  ]


def test_continuous_eval_stops_once_all_evaluators_failed():
  trainer = _make_trainer({
      'broken': RuntimeError('broken eval'),
      'flaky': [None, RuntimeError('flaky eval')],  # Fails on the 2nd call.
  })

  with pytest.raises(ExceptionGroup, match='All evaluators') as exc_info:
    _continuous_eval(trainer, eval_names=['broken', 'flaky'], steps=[0, 1, 2])

  # `broken` is skipped after step 0 and the job stops when `flaky` fails on
  # step 1, so step 2 is never evaluated.
  assert len(exc_info.value.exceptions) == 2
  assert trainer.evals['broken'].evaluate.call_count == 1
  assert trainer.evals['flaky'].evaluate.call_count == 2
