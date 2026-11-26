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

"""Tests for schedule plotting helpers."""

import re

import altair as alt
from kauldron import inspect
import optax
import pytest


def test_plot_schedules_constant_schedule():
  """Verifies constant schedules render valid HTML on the default Altair."""
  fig = inspect.plot_schedules(
      {'learning_rate': optax.constant_schedule(0.01)},
      num_steps=100,
  )

  spec = fig.to_dict()
  assert len(spec['layer']) == 2
  html = fig.to_html()
  assert 'learning_rate' in html


def test_plot_schedules_nested_dict():
  """Verifies nested schedule dicts flatten into dotted series names."""
  schedules = {
      'opt': {
          'lr': optax.constant_schedule(1e-3),
          'wd': optax.linear_schedule(0.0, 0.1, transition_steps=100),
      },
  }

  fig = inspect.plot_schedules(schedules, num_steps=100)

  html = fig.to_html()
  assert 'opt.lr' in html
  assert 'opt.wd' in html


@pytest.mark.parametrize('version', ['5.5.0', '6.0.0rc1', '6.3.0', '7.0.0'])
def test_plot_schedules_supports_altair_v5_and_newer(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
):
  """Verifies Altair 5+ major, minor, and pre-release versions are accepted."""
  monkeypatch.setattr(alt, '__version__', version)

  fig = inspect.plot_schedules(
      {'lr': optax.constant_schedule(0.1)},
      num_steps=10,
  )

  assert len(fig.to_dict()['layer']) == 2


def test_plot_schedules_supports_altair_v5_without_when(
    monkeypatch: pytest.MonkeyPatch,
):
  """Verifies Altair 5.0-5.3 without alt.when falls back to alt.condition."""
  monkeypatch.setattr(alt, '__version__', '5.0.0')
  monkeypatch.delattr(alt, 'when')

  fig = inspect.plot_schedules(
      {'lr': optax.constant_schedule(0.1)},
      num_steps=10,
  )

  assert len(fig.to_dict()['layer']) == 2


@pytest.mark.parametrize('version', ['3.2.0', '4.1.0', 'unknown'])
def test_plot_schedules_rejects_unsupported_altair_version(
    monkeypatch: pytest.MonkeyPatch,
    version: str,
):
  """Verifies unsupported or malformed Altair versions raise ValueError."""
  monkeypatch.setattr(alt, '__version__', version)

  with pytest.raises(
      ValueError, match=rf'Unsupported Altair version: {re.escape(version)}'
  ):
    inspect.plot_schedules(
        {'lr': optax.constant_schedule(0.1)},
        num_steps=10,
    )
