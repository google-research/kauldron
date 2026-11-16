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

"""Test utils."""

import os
import sys

import jax


def assert_trees_all_same_type(a, b) -> None:
  """Asserts that all leafs from `a` and `b` are of the same type."""

  def _assert_same_type(a, b):
    assert type(a) == type(b)  # pylint: disable=unidiomatic-typecheck

  jax.tree.map(_assert_same_type, a, b)


def require_two_devices() -> None:
  """Skips unless `jax.local_device_count() == 2` (fails if 2 requested)."""
  device_count = jax.local_device_count()
  if device_count == 2:
    return

  pytest = sys.modules.get('pytest')
  xla_flags = os.environ.get('XLA_FLAGS', '').split()
  requested_two = (
      any(
          tok.lstrip('-') == 'xla_force_host_platform_device_count=2'
          for tok in xla_flags
      )
      or os.environ.get('JAX_NUM_CPU_DEVICES') == '2'
  )
  if requested_two:
    msg = (
        'Expected 2 local devices via XLA_FLAGS or JAX_NUM_CPU_DEVICES=2,'
        f' but JAX initialized with {device_count}.'
    )
    if pytest is not None:
      pytest.fail(msg)
    raise AssertionError(msg)

  msg = f'Requires exactly 2 local JAX devices, got {device_count}'
  if pytest is not None:
    pytest.skip(msg)
  raise RuntimeError(msg)
