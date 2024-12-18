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

"""Freeze utils."""

from collections.abc import Callable
import functools
from typing import Any

import jax
import optax

_PyTree = Any


def partial_updates(
    optimizer: optax.GradientTransformation,
    mask: _PyTree | Callable[[_PyTree], _PyTree],
) -> optax.GradientTransformation:
  """Applies the optimizer to a subset of the parameters.

  Args:
    optimizer: The optimizer to use.
    mask: A tree or callable returning a tree of bools to apply the optimizer
      to.

  Returns:
    The wrapped optimizer.
  """

  return optax.multi_transform(
      {
          'train': optimizer,
          'freeze': optax.set_to_zero(),
      },
      functools.partial(_make_labels, mask=mask),
  )


def _make_labels(tree, mask):
  if callable(mask):
    mask = mask(tree)
  return jax.tree.map(lambda x: 'train' if x else 'freeze', mask)
