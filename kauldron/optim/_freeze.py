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

import functools

import jax
import optax


def partial_updates(optimizer, mask):
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
  tree = jax.tree.map(lambda x: 'train' if mask else 'freeze', tree)
  return tree
